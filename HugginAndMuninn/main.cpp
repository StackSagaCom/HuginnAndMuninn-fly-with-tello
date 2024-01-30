#define TELLO_DEBUG     // This can be used to enable verbose logging
#include <iostream>
#include "../../tello.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <chrono>
#include <thread>
using namespace cv;
using namespace std;

void moveTello(Tello& tello) {
	const float distance = 90.0;
	tello.takeoff();
	tello.move_forward(distance);
	tello.move_back(distance);
	tello.land();
}

void recordVideo(cv::VideoCapture& capture, cv::VideoWriter& videoWriter, const int durationInSeconds) {
	auto startTime = std::chrono::steady_clock::now();

	while (true) {
		if (!capture.grab()) {
			std::cerr << "Error: Failed to grab frame" << std::endl;
			break;
		}

		cv::Mat frame;
		capture.retrieve(frame);

		if (frame.empty()) {
			std::cerr << "Error: Failed to capture frame" << std::endl;
			break;
		}

		videoWriter.write(frame);

		auto currentTime = std::chrono::steady_clock::now();
		auto elapsedTime = std::chrono::duration_cast<std::chrono::seconds>(currentTime - startTime).count();

		if (elapsedTime >= durationInSeconds) {
			break;
		}
	}

	videoWriter.release();
}

void analyzeStoredVideo(const std::string& videoPath) {
	cv::VideoCapture capture(videoPath);

	if (!capture.isOpened()) {
		std::cerr << "Error: Unable to open video file " << videoPath << std::endl;
		return;
	}

	cv::CascadeClassifier faceCascade;
	faceCascade.load("./FaceDetection.xml");

	if (faceCascade.empty()) {
		std::cerr << "Error: XML file not loaded" << std::endl;
		return;
	}

	 while (true) {
        cv::Mat frame;
        capture >> frame;

        if (frame.empty()) {
            break;  // End of video
        }

        std::vector<cv::Rect> faces;
        faceCascade.detectMultiScale(frame, faces, 1.1, 10);

        for (const auto& face : faces) {
            cv::rectangle(frame, face.tl(), face.br(), cv::Scalar(255, 0, 255), 3);
        }

        cv::imshow("Video Analysis", frame);
		cv::waitKey(1);
    }
}

int main() {
	std::this_thread::sleep_for(std::chrono::seconds(5));
	Tello tello;

	if (!tello.connect()) {
		std::cerr << "Error: Unable to connect to Tello" << std::endl;
		return 1;
	}

	tello.enable_video_stream();

	cv::VideoCapture capture{"udp://0.0.0.0:11111?overrun_nonfatal=1&fifo_size=580000", cv::CAP_FFMPEG};
	if (!capture.isOpened()) {
		std::cerr << "Error: Unable to open video stream" << std::endl;
		return 1;
	}

	capture.set(cv::CAP_PROP_FPS, 30);
	capture.set(cv::CAP_PROP_BUFFERSIZE, 10);

	int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
	int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
	cv::VideoWriter videoWriter("output_video.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(width, height));

	const int durationInSeconds = 20;

	// Launch threads for Tello movements and video recording
	std::thread telloThread(moveTello, std::ref(tello));
	std::thread videoThread(recordVideo, std::ref(capture), std::ref(videoWriter), durationInSeconds);

	// Wait for both threads to finish
	telloThread.join();
	videoThread.join();



	std::string videoPath = "";  // Change this to your video file path to output_video.avi
	analyzeStoredVideo(videoPath);
	PRINTF_INFO("[Press Enter to exit]");
	std::cin.get();
	return 0;
}
