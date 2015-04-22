/*
 * utils.h
 *
 *  Created on: Mar 6, 2015
 *      Author: teider
 */
#ifndef UTILS_H_
#define UTILS_H_

#include <string>
#include <vector>

void cleanOutputDir(const char *dir);

std::vector<std::string> getImagesList(const char *dir);
std::vector<float> generateLabels(std::vector<std::string> file_list);


#endif /* UTILS_H_ */
