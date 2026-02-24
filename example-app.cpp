#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <span>
#include <tuple>
#include <map>
#include <algorithm>

int main() {

  std::ifstream file("../names.txt");

  if (!file.is_open()) {
      std::cerr << "Error opening file\n";
      return 1;
  }

  // load names
  std::vector<std::string> lines;
  for (std::string line; std::getline(file, line); ) {
      lines.push_back(line);
  }
  file.close();

  std::map<std::tuple<char, char>, int> b;
  torch::Tensor N = torch::zeros({ 27, 27 });
  for (std::string word : lines) {
    word = "." + word + ".";

    for (size_t i = 0; i < word.size() - 1; i++) {
      char ch1 = word[i];
      char ch2 = word[i+1];

      auto ctoi = [](char ch) {
        if (ch == '.') return 0;
        return ch - 'a' + 1;
      };

      // std::cout << "ch1 " << ch1 << " int: " << ctoi(ch1) << std::endl;

      N.index_put_({ ctoi(ch1), ctoi(ch2) }, N.index({ ctoi(ch1), ctoi(ch2) }).add_(1));

      std::cout << "ch1: " << ch1 << ", ch2: " << ch2 << std::endl;
    }
  }

  // for (auto it : b) {
  //   std::cout << "(" << get<0>(it.first) << ", " << get<1>(it.first) << ")" << " : " << it.second << std::endl;
  // }

  // using bigramTuple = std::pair<std::tuple<char, char>, int>;
  // std::vector<bigramTuple> arr(b.begin(), b.end());

  // std::sort(arr.begin(), arr.end(), [](bigramTuple a, bigramTuple b) {
  //   return a.second > b.second;
  // });

  // for (auto p : arr) {
  //   auto [a, b] = p; 
  //   std::cout << "(" << get<0>(a) << ", " << get<1>(a) << ")" << " : " << b << std::endl;
  // }

  std::cout << N << std::endl;

}