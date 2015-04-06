#include "hog.h"
#include "hog_constants.h"

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "utils.h"

#define INPUT_PATH "resources/images"

//#define DEBUG

int main(int argc, char **argv) {

  std::vector<std::string> image_list = getImagesList(INPUT_PATH);

  cv::Mat img;
  cv::Mat out;

  for (int i = 0; i < image_list.size(); ++i) {
    std::cout << image_list[i] << std::endl;
    img = cv::imread(image_list[i], CV_LOAD_IMAGE_GRAYSCALE);
    img.convertTo(img, CV_32FC1);
    out = draw_hog(calc_hog(img));
    cv::imwrite("outputs/" + image_list[i], out);
    //break;
  }
}

hog calc_hog(cv::Mat img) {
  hog ret;

  cv::Mat img_dx(img.rows, img.cols, img.type());
  cv::Mat img_dy(img.rows, img.cols, img.type());

  cv::Mat img_mag(img.rows, img.cols, img.type());
  cv::Mat img_angle(img.rows, img.cols, img.type());

  cv::Mat dx = (cv::Mat_<float>(3,3) << 0, 0, 0, -1, 0, 1, 0, 0, 0);
  cv::Mat dy = (cv::Mat_<float>(3,3) << 0, -1, 0, 0, 0, 0, 0, 1, 0);

  cv::filter2D(img, img_dx, -1, dx, cv::Point(-1,-1), 0.0, cv::BORDER_REPLICATE);
  cv::filter2D(img, img_dy, -1, dy, cv::Point(-1,-1), 0.0, cv::BORDER_REPLICATE);

  cv::cartToPolar(img_dx, img_dy, img_mag, img_angle, true);

  float bin_size = 360.0 / HOG_NUMBER_OF_BINS;

  int n_cells = HOG_NUMBER_OF_BLOCKS * HOG_CELLS_PER_BLOCK;

  int top_row = 0;
  int bottom_row = (img.rows / HOG_GRID_HEIGHT) - 1;
  int extra_rows = (img.rows % HOG_GRID_HEIGHT);

  int left_col = 0;
  int right_col = (img.cols / HOG_GRID_WIDTH) - 1;
  int extra_cols = (img.cols % HOG_GRID_WIDTH);

  for (int i = 0; i < n_cells; ++i) {
    histogram hist;
    for (int j = 0; j < HOG_NUMBER_OF_BINS; ++j) {
      hist.bins.push_back(0.0f);
    }

    if (extra_rows) {
      bottom_row++;
      extra_rows--;
    }
    if (extra_cols) {
      right_col++;
      extra_cols--;
    }

    for (int j = top_row; j <= bottom_row; ++j) {
      for (int k = left_col; k <= right_col; ++k) {
        if (img_mag.at<float>(j,k) > 0) {
          int left_bin = (int)floor(((img_angle.at<float>(j,k) - (bin_size / 2)) / bin_size));
          if (left_bin < 0) left_bin += hist.bins.size();
          int right_bin = (left_bin + 1) % hist.bins.size();

          float delta = (img_angle.at<float>(j,k) / bin_size) - right_bin;
          if (delta > 1.0) delta -= hist.bins.size();

          hist.bins[left_bin] += (0.5 - delta) * img_mag.at<float>(j,k);
          hist.bins[right_bin] += (0.5 + delta) * img_mag.at<float>(j,k);

#ifdef DEBUG
          std::cout << "Processing pixel (" << j << ", " << k << "):" << std::endl;
          std::cout << "Original value: " << img.at<float>(j,k) << std::endl;
          std::cout << "dx: " << img_dx.at<float>(j,k) << "   ";
          std::cout << "dy: " << img_dy.at<float>(j,k) << std::endl;
          std::cout << "Magnitude: " << img_mag.at<float>(j,k) << "   ";
          std::cout << "Angle: " << img_angle.at<float>(j,k) << std::endl;
          std::cout << "Left bin: " << left_bin << "     Right bin: " << right_bin << "   ";
          std::cout << "Delta: " << delta << std::endl;

          for (int l = 0; l < hist.bins.size(); ++l) {
            std::cout << "hist[" << l << "]: " << hist.bins[l] << "   ";
          }
          std::cout << std::endl;
          std::cout << std::endl;
#endif //DEBUG
        }
      }
    }

    float max_bin = 0.0f;

    for (int i = 0; i < hist.bins.size(); ++i) {
      if (max_bin < hist.bins[i]) max_bin = hist.bins[i];
    }
    for (int i = 0; i < hist.bins.size(); ++i) {
      hist.bins[i] /= max_bin;
      std::cout << "hist[" << i << "]: " << hist.bins[i] << "   ";
    }
    std::cout << std::endl;
    std::cout << std::endl;

    ret.cells.push_back(hist);
  }

  return ret;
}

cv::Mat draw_hog(hog h) {
  cv::Mat ret(HOG_DRAW_IMAGE_HEIGHT, HOG_DRAW_IMAGE_WIDTH, CV_32F);

  int cell_height = ret.rows / h.cells.size();
  int bin_width = ret.cols / HOG_NUMBER_OF_BINS;
  float pixel_hvalue = (1 / cell_height);

  for (int i = 0; i < h.cells.size(); ++i) {
    for (int j = 0; j < h.cells[i].bins.size(); ++j) {
      if (h.cells[i].bins[j] > 0) {

      }
    }
  }
  return ret;
}















