#include <SLAMBenchAPI.h>

#include <io/SLAMFrame.h>
#include <io/sensor/DepthSensor.h>
#include <io/sensor/CameraSensor.h>
#include <values/Value.h>

#include <chrono>
#include <thread>

#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <cassert>

#include <cnn_interface/CaffeInterface.h>
#include <map_interface/ElasticFusionInterface.h>
#include <semantic_fusion/SemanticFusionInterface.h>
#include <utilities/LiveLogReader.h>
#include <utilities/RawLogReader.h>
#include <utilities/PNGLogReader.h>
#include <utilities/Types.h>

#include <gui/Gui.h>

std::vector<ClassColour> load_colour_scheme(std::string filename, int num_classes) {
    std::vector<ClassColour> colour_scheme(num_classes);
    std::ifstream file(filename);
    std::string str; 
    int line_number = 1;
    while (std::getline(file, str)) {
        std::istringstream buffer(str);
        std::string textual_prefix;
        int id, r, g, b;
        if (line_number > 2) {
            buffer >> textual_prefix >> id >> r >> g >> b;
            ClassColour class_colour(textual_prefix,r,g,b);
            assert(id < num_classes);
            colour_scheme[id] = class_colour;
        }
        line_number++;
    }
    return colour_scheme;
}

// Until c++14 is here, this will have to do
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

slambench::outputs::Output *pose_output        = nullptr;
slambench::outputs::Output *rgb_frame_output   = nullptr;
slambench::outputs::Output *depth_frame_output = nullptr;
slambench::outputs::Output *pointcloud_output  = nullptr;

slambench::io::DepthSensor  *depth_sensor;
slambench::io::CameraSensor *rgb_sensor;

uint8_t*  inputRGB      = nullptr;
uint16_t* inputDepth    = nullptr;
uint8_t*  renderedDepth = nullptr;

uint64_t  timestamp     = 0;

// CNN Skip params
const int cnn_skip_frames = 10;
// Option CPU-based CRF smoothing
const bool use_crf        = false;
const int crf_skip_frames = 500;
const int crf_iterations  = 10;

CaffeInterface caffeInterface;

// TODO: find better names
const std::string file1 = "nyu_rgbd/inference.prototxt";
const std::string file2 = "nyu_rgbd/inference.caffemodel";

const std::string colorscheme = "class_colour_scheme.data";

std::vector<ClassColour> class_colour_lookup;
std::unique_ptr<SemanticFusionInterface> semantic_fusion;

std::unique_ptr<Gui> gui;
std::unique_ptr<ElasticFusionInterface> map;

bool sb_new_slam_configuration(SLAMBenchLibraryHelper *slam_settings) {
    return true;
}

void initAlgorithm() {

    if (rgb_sensor == nullptr || depth_sensor == nullptr) {
        std::cerr << "The sensors must be initialized before the algorithm. Aborting." << std::endl;
        abort();
    }

    caffeInterface.Init(file1, file2);

    class_colour_lookup = load_colour_scheme("class_colour_scheme.data", caffeInterface.num_output_classes());

    semantic_fusion = make_unique<SemanticFusionInterface>(caffeInterface.num_output_classes(), 100);

    Resolution::getInstance(rgb_sensor->Width, rgb_sensor->Height);
    Intrinsics::getInstance(528, 528, 319, 240);

    gui = make_unique<Gui>(true, class_colour_lookup, 640, 480, true);
    map = make_unique<ElasticFusionInterface>();
    if (!map->Init(class_colour_lookup)) {
      std::cout<<"ElasticFusionInterface init failure"<<std::endl;
    }

}

bool sb_init_slam_system(SLAMBenchLibraryHelper * slam_settings) {

    for(const auto &sensor : slam_settings->get_sensors()) {
        if (sensor->GetType() == "Camera" && !rgb_sensor) {
            rgb_sensor = dynamic_cast<slambench::io::CameraSensor*>(sensor);
        } else if(sensor->GetType() == "Depth") {
            depth_sensor = dynamic_cast<slambench::io::DepthSensor*>(sensor);
        }
    }

    if (rgb_sensor->Width != depth_sensor->Width || rgb_sensor->Height != depth_sensor->Height) {
        std::cerr << "ERROR: The RGB and depth sensor sizes do not match. Aborting." << std::endl;
        abort();
    }

    inputRGB      = new unsigned char[rgb_sensor->Width * rgb_sensor->Height * 3];
    inputDepth    = new uint16_t[depth_sensor->Width * depth_sensor->Height];
    renderedDepth = new uint8_t[depth_sensor->Width * depth_sensor->Height];

    initAlgorithm();

    pose_output = new slambench::outputs::Output("Pose", slambench::values::VT_POSE, true);
    slam_settings->GetOutputManager().RegisterOutput(pose_output);
    pose_output->SetActive(true);

    pointcloud_output = new slambench::outputs::Output("PointCloud", slambench::values::VT_COLOUREDPOINTCLOUD, true);
    pointcloud_output->SetKeepOnlyMostRecent(true);
    slam_settings->GetOutputManager().RegisterOutput(pointcloud_output);

    rgb_frame_output = new slambench::outputs::Output("RGB Frame", slambench::values::VT_FRAME);
    rgb_frame_output->SetKeepOnlyMostRecent(true);
    slam_settings->GetOutputManager().RegisterOutput(rgb_frame_output);
    rgb_frame_output->SetActive(true);

    depth_frame_output = new slambench::outputs::Output("Depth Frame", slambench::values::VT_FRAME);
    depth_frame_output->SetKeepOnlyMostRecent(true);
    slam_settings->GetOutputManager().RegisterOutput(depth_frame_output);
    depth_frame_output->SetActive(true);

    return true;
}


bool sb_update_frame (SLAMBenchLibraryHelper * slam_settings, slambench::io::SLAMFrame* s) {

    static bool depth_ready = false;
    static bool rgb_ready   = false;

    void *target      = nullptr;
    bool  renderDepth = false;

    if (s->FrameSensor == depth_sensor) {
        target = inputDepth;
        depth_ready = true;
        renderDepth = true;
    } else if (s->FrameSensor == rgb_sensor) {
        target = inputRGB;
        rgb_ready = true;
    } else {
    }

    timestamp = static_cast<int64_t>(s->Timestamp.S) * 1000 + s->Timestamp.Ns / 1000;

    if (target != nullptr) {
        memcpy(target, s->GetData(), s->GetSize());
        s->FreeData();
    }

    if (renderDepth) {
        for (size_t index = 0; index < depth_sensor->Width * depth_sensor->Height; index++) {
            renderedDepth[index] = inputDepth[index] >> 4;
        }
    }

    if (depth_ready && rgb_ready) {
        depth_ready = false;
        rgb_ready = false;
        return true;
    } else {
        return false;
    }
}

std::shared_ptr<caffe::Blob<float>> segmented_prob;
int frame_num = 0;

bool sb_process_once(SLAMBenchLibraryHelper * slam_settings) {

    const int num_classes = caffeInterface.num_output_classes();

    if (!map->ProcessFrame(inputRGB, inputDepth, timestamp)) {
        std::cout<<"Elastic fusion lost!"<<std::endl;
    }

    // This queries the map interface to update the indexes within the table 
    // It MUST be done everytime ProcessFrame is performed as long as the map
    // is not performing tracking only (i.e. fine to not call, when working
    // with a static map)
    semantic_fusion->UpdateProbabilityTable(map);

    // We do not need to perform a CNN update every frame, we perform it every
    // 'cnn_skip_frames'
    if (frame_num == 0 || (frame_num > 1 && ((frame_num + 1) % cnn_skip_frames == 0))) {
        segmented_prob = caffeInterface.ProcessFrame(inputRGB, inputDepth,
                                                     rgb_sensor->Height, rgb_sensor->Width);
        semantic_fusion->UpdateProbabilities(segmented_prob,map);
    }

    if (use_crf && frame_num % crf_skip_frames == 0) {
        std::cout<<"Performing CRF Update..."<<std::endl;
        semantic_fusion->CRFUpdate(map,crf_iterations);
    }

    frame_num++;
    return true;
}


bool sb_get_tracked(bool* tracked) {
    return true;
}

bool sb_clean_slam_system() {
    return true;
}

bool sb_update_outputs(SLAMBenchLibraryHelper *lib, const slambench::TimeStamp *latest_output) {

    // The SemanticFusion GUI
    gui->preCall();
    gui->renderMap(map);
    gui->displayRawNetworkPredictions("pred",segmented_prob->mutable_gpu_data());
    // This is to display a predicted semantic segmentation from the fused map
    semantic_fusion->CalculateProjectedProbabilityMap(map);
    gui->displayArgMaxClassColouring("segmentation",semantic_fusion->get_rendered_probability()->mutable_gpu_data(),
                                     caffeInterface.num_output_classes(),
                                     semantic_fusion->get_class_max_gpu()->gpu_data(),
                                     semantic_fusion->max_num_components(),map->GetSurfelIdsGpu(),0.0);
    gui->postCall();


    // The SLAMBench outputs
    slambench::TimeStamp ts = *latest_output;

    if (pose_output->IsActive()) {
        std::lock_guard<FastLock> lock (lib->GetOutputManager().GetLock());
        pose_output->AddPoint(ts, new slambench::values::PoseValue(map->getCurrentPose()));
    }

    if (rgb_frame_output->IsActive()) {
        std::lock_guard<FastLock> lock (lib->GetOutputManager().GetLock());

        rgb_frame_output->AddPoint(*latest_output,
                        new slambench::values::FrameValue(rgb_sensor->Width, rgb_sensor->Height,
                        slambench::io::pixelformat::EPixelFormat::RGB_III_888 , inputRGB));
    }

    if (depth_frame_output->IsActive()) {
        std::lock_guard<FastLock> lock (lib->GetOutputManager().GetLock());

        auto frameValue = new slambench::values::FrameValue(depth_sensor->Width, depth_sensor->Height,
                              slambench::io::pixelformat::EPixelFormat::G_I_8, renderedDepth);
 

        depth_frame_output->AddPoint(*latest_output, frameValue);
    }


    if (pointcloud_output->IsActive()) {
        slambench::values::ColoredPointCloudValue *point_cloud = new slambench::values::ColoredPointCloudValue();

        Eigen::Vector4f * mapData = map->downloadMap();

        for(unsigned int i = 0; i < map->getLastCount(); i++) {

            Eigen::Vector4f pos = mapData[(i * 3) + 0];
            Eigen::Vector4f col = mapData[(i * 3) + 1];

            slambench::values::ColoredPoint3DF new_vertex(pos[0], pos[1], pos[2]);
            new_vertex.R = int(col[0]) >> 16 & 0xFF;
            new_vertex.G = int(col[0]) >>  8 & 0xFF;
            new_vertex.B = int(col[0])       & 0xFF;

            int r = new_vertex.R;
            int g = new_vertex.G;
            int b = new_vertex.B;

            point_cloud->AddPoint(new_vertex);
        }

        // we're finished with the map data we got from efusion, so delete it
        delete mapData;

        // Take lock only after generating the map
        std::lock_guard<FastLock> lock (lib->GetOutputManager().GetLock());

        pointcloud_output->AddPoint(ts, point_cloud);
    }

    return true;
}
