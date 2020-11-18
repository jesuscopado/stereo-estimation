#include <opencv2/opencv.hpp>
#include <iostream>
#include <string> 
#include <fstream>
#include <sstream>
#include <cmath>
#include <omp.h>
#include "main.h"

int main(int argc, char** argv) {

  ////////////////
  // Parameters //
  ////////////////

  // camera setup parameters
  const double focal_length = 3740;
  const double baseline = 160;

  // stereo estimation parameters
  const int dmin = 200;
  const double scale = 3;

  ///////////////////////////
  // Commandline arguments //
  ///////////////////////////

  if (argc < 7) {
    std::cerr << "Usage: " << argv[0] <<
    " IMAGE1 IMAGE2 OUTPUT_FILE WINDOW_SIZE_NAIVE WINDOW_SIZE_DYNAMIC WEIGHT" << std::endl;
    return 1;
  }

  cv::Mat image1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
  cv::Mat image2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
  const std::string output_file = argv[3];
  const int window_size_naive = atoi(argv[4]);
  const int window_size_dynamic = atoi(argv[5]);
  const int weight = atoi(argv[6]);

  if (!image1.data) {
    std::cerr << "No image1 data" << std::endl;
    return EXIT_FAILURE;
  }

  if (!image2.data) {
    std::cerr << "No image2 data" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "------------------ Parameters -------------------" << std::endl;
  std::cout << "focal_length = " << focal_length << std::endl;
  std::cout << "baseline = " << baseline << std::endl;
  std::cout << "window_size_naive = " << window_size_naive << std::endl;
  std::cout << "window_size_dynamic = " << window_size_dynamic << std::endl;
  std::cout << "occlusion weights = " << weight << std::endl;
  std::cout << "disparity added due to image cropping = " << dmin << std::endl;
  std::cout << "scaling of disparity images to show = " << scale << std::endl;
  std::cout << "output filename = " << argv[3] << std::endl;
  std::cout << "-------------------------------------------------" << std::endl;

  int height = image1.size().height;
  int width = image1.size().width;

  ///////////////////////////////
  // Reconstruction and output //
  ///////////////////////////////

  /// Naive approach

  // naive disparity image
  cv::Mat naive_disparities = cv::Mat::zeros(height, width, CV_8UC1);

  // stereo estimation
  StereoEstimation_Naive(
    window_size_naive, dmin, height, width,
    image1, image2,
    naive_disparities, scale);

  // reconstruction
  Disparity2PointCloud(
     output_file + "_naive",
     height, width, naive_disparities,
     window_size_naive, dmin, baseline, focal_length);

  // save / display images
  std::stringstream out1;
  out1 << output_file << "_naive.png";
  cv::imwrite(out1.str(), naive_disparities);

  cv::namedWindow("Naive", cv::WINDOW_AUTOSIZE);
  cv::imshow("Naive", naive_disparities);

  /// Dynamic programming approach

  // dynamic disparity image
  cv::Mat dynamic_disparities = cv::Mat::zeros(height, width, CV_8UC1);

  // stereo estimation
  StereoEstimation_Dynamic(
    window_size_dynamic, height, width, weight,
    image1, image2,
    dynamic_disparities, scale);

  // reconstruction
  Disparity2PointCloud(
    output_file + "_dynamic",
    height, width, dynamic_disparities,
    window_size_dynamic, dmin, baseline, focal_length);

  // save / display images
  std::stringstream out2;
  out2 << output_file << "_dynamic.png";

  cv::Mat dynamic_disparities_image;
  cv::normalize(dynamic_disparities, dynamic_disparities_image, 255, 0, cv::NORM_MINMAX);

  cv::imwrite(out2.str(), dynamic_disparities_image);

  cv::namedWindow("Dynamic", cv::WINDOW_AUTOSIZE);
  cv::imshow("Dynamic", dynamic_disparities_image);

  /// OpenCV implementation

  cv::Mat opencv_disparities;
  cv::Ptr<cv::StereoBM > match = cv::StereoBM::create(16, 9);
  match->compute(image1, image2, opencv_disparities);
  cv::imshow("OpenCV result",opencv_disparities*1000);

  std::stringstream out3;
  out3 << output_file << "_opencv.png";
  cv::imwrite(out3.str(), opencv_disparities);

  cv::waitKey(0);

  return 0;
}

void StereoEstimation_Naive(
  const int& window_size,
  const int& dmin,
  int height,
  int width,
  cv::Mat& image1, cv::Mat& image2, cv::Mat& naive_disparities, const double& scale)
{
  int half_window_size = window_size / 2;

  for (int i = half_window_size; i < height - half_window_size; ++i) {

    std::cout
      << "Calculating disparities for the naive approach... "
      << std::ceil(((i - half_window_size + 1) / static_cast<double>(height - window_size + 1)) * 100) << "%\r"
      << std::flush;

    for (int j = half_window_size; j < width - half_window_size; ++j) {
      int min_ssd = INT_MAX;
      int disparity = 0;

      for (int d = -j + half_window_size; d < width - j - half_window_size; ++d) {
        int ssd = 0;

        for (int u = -half_window_size; u <= half_window_size; ++u) {
            for (int v = -half_window_size; v <= half_window_size; ++v) {
                int val_left = image1.at<uchar>(i + u, j + v);
                int val_right = image2.at<uchar>(i + u, j + v + d);
                ssd += (val_left - val_right) * (val_left - val_right);
            }
        }

        // TODO: improve this approach by substracting the ssd of the column that gets out
        //  of the window by the shift and adding the ssd of the new column that gets in

        if (ssd < min_ssd) {
          min_ssd = ssd;
          disparity = d;
        }
      }

      naive_disparities.at<uchar>(i - half_window_size, j - half_window_size) = std::abs(disparity) * scale;
    }
  }

  std::cout << "Calculating disparities for the naive approach... Done.\r" << std::flush;
  std::cout << std::endl;
}

void StereoEstimation_Dynamic(
        const int& window_size,
        int height,
        int width,
        int weight,
        cv::Mat& image1, cv::Mat& image2,
        cv::Mat& dynamic_disparities, const double& scale)
{
    int half_window_size = window_size / 2;

    for (int r = half_window_size; r < height - half_window_size; ++r) {

        std::cout
                << "Calculating disparities for the dynamic approach... "
                << std::ceil(((r - half_window_size + 1) / static_cast<double>(height - window_size + 1)) * 100) << "%\r"
                << std::flush;

        cv::Mat C = cv::Mat::zeros(width, width, CV_32F);
        cv::Mat M = cv::Mat::zeros(width, width, CV_8UC1);

        for (int x = 1; x < width; ++x) {
            C.at<float>(x, 0) = x * weight;
            M.at<uchar>(x, 0) = 3;
        }

        for (int y = 1; y < width; ++y) {
            C.at<float>(0, y) = y * weight;
            M.at<uchar>(0, y) = 2;
        }

        for (int x = 1; x < width; ++x) {
            for (int y = 1; y < width; ++y) {

                double d = DisparitySpaceImage(image1, image2, half_window_size, r, x, y);
                double match_cost = C.at<float>(x-1, y-1) + d;
                double left_occl_cost = C.at<float>(x-1, y) + weight;
                double right_occl_cost = C.at<float>(x, y-1) + weight;

                if (match_cost < std::min(left_occl_cost, right_occl_cost)) {
                    C.at<float>(x, y) = match_cost;
                    M.at<uchar>(x, y) = 1;
                }
                else if (left_occl_cost < std::min(match_cost, right_occl_cost)) {
                    C.at<float>(x, y) = left_occl_cost;
                    M.at<uchar>(x, y) = 2;
                }
                else { // (right_occl_cost < std::min(match_cost, left_occl_cost))
                    C.at<float>(x, y) = right_occl_cost;
                    M.at<uchar>(x, y) = 3;
                }

            }
        }

        int x = width - 1;
        int y = width - 1;
        int c = width;
        int d = 0;
        while (x != 0 && y != 0) {
            switch (M.at<uchar>(x, y)) {
                case 1:
                    d = abs(x - y);
                    x--;
                    y--;
                    c--;
                    break;
                case 2:
                    x--;
                    break;
                case 3:
                    y--;
                    break;
            }
            dynamic_disparities.at<uchar>(r - half_window_size, c) = d;
        }
    }

    std::cout << "Calculating disparities for the dynamic approach... Done.\r" << std::flush;
    std::cout << std::endl;
}

int DisparitySpaceImage(
        cv::Mat& image1, cv::Mat& image2,
        int half_window_size, int r, int x, int y)
{
    int ssd = 0;
    for (int u = -half_window_size; u <= half_window_size; ++u) {
        for (int v = -half_window_size; v <= half_window_size; ++v) {
            int val_left = image1.at<uchar>(r + u, x + v);
            int val_right = image2.at<uchar>(r + u, y + v);
            ssd += (val_left - val_right) * (val_left - val_right);
        }
    }
    return ssd;
}


void Disparity2PointCloud(
  const std::string& output_file,
  int height, int width, cv::Mat& disparities,
  const int& window_size,
  const int& dmin, const double& baseline, const double& focal_length)
{
  std::stringstream out3d;
  out3d << output_file << ".xyz";
  std::ofstream outfile(out3d.str());
  for (int i = 0; i < height - window_size; ++i) {
    std::cout << "Reconstructing 3D point cloud from disparities... " << std::ceil(((i) / static_cast<double>(height - window_size + 1)) * 100) << "%\r" << std::flush;
    for (int j = 0; j < width - window_size; ++j) {
      if (disparities.at<uchar>(i, j) == 0) continue;

      const double Z = focal_length * baseline / (disparities.at<uchar>(i, j) + dmin);
      const double X = (i - width / 2) * Z / focal_length;
      const double Y = (j - height / 2) * Z / focal_length;
      outfile << X << " " << Y << " " << Z << std::endl;
    }
  }

  std::cout << "Reconstructing 3D point cloud from disparities... Done.\r" << std::flush;
  std::cout << std::endl;
}
