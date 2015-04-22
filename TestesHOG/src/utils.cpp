#include "utils.h"

#include <iostream>

#include <boost/filesystem.hpp>

void cleanOutputDir(const char *dir_path) {
  boost::filesystem::path dir(dir_path);

  if (exists(dir)) {
    remove_all(dir);
  }
}

std::vector<std::string> getImagesList(const char *dir_path) {

  std::vector<std::string> ret;

  boost::filesystem::path dir(dir_path);

  if (!exists(dir)) {
    std::cout << "O diretório de entrada não existe." << std::endl;
    return ret;
  }

  boost::filesystem::recursive_directory_iterator dir_it(dir);
  boost::filesystem::recursive_directory_iterator end_it;

  while (dir_it != end_it) {
    if (is_regular_file((*dir_it).path())) {
      ret.push_back((*dir_it).path().string());
    }
    ++dir_it;
  }
  return ret;
}

std::vector<float> generateLabels(std::vector<std::string> file_list) {
  std::vector<float> ret;

  for(int i = 0; i < file_list.size(); ++i) {
    //If it isn't negative, it's positive
    if(file_list[i].find("negative") == std::string::npos) {
      ret.push_back(1.0);
    } else {
      ret.push_back(-1.0);
    }
  }

  return ret;
}
