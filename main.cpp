#include <torch/torch.h>
#include <ATen/CPUGeneratorImpl.h>

#include <iostream>
#include <fstream>
#include <span>
#include <tuple>
#include <map>
#include <algorithm>

#include <TCanvas.h>
#include <TH2I.h>
#include <TStyle.h>

void draw_bigram_root(torch::Tensor N) {
  const int ALPHABET_SIZE = 27;
  gStyle->SetOptStat(0);
  gStyle->SetPaintTextFormat("g"); 

  N = N.to(torch::kCPU).contiguous();

  auto h = new TH2I("h", "Bigram counts;first letter;second letter",
                    ALPHABET_SIZE, 0, ALPHABET_SIZE,
                    ALPHABET_SIZE, 0, ALPHABET_SIZE);

  for (int i = 0; i < ALPHABET_SIZE; ++i) {
    TString lab; lab += char('a' + i);
    if (i == ALPHABET_SIZE - 1) { lab = '.'; }
    h->GetXaxis()->SetBinLabel(i + 1, lab);
    h->GetYaxis()->SetBinLabel(i + 1, lab);
  }

  auto acc = N.accessor<int64_t, 2>();
  for (int i = 0; i < ALPHABET_SIZE; ++i) {
    for (int j = 0; j < ALPHABET_SIZE; ++j) {
      h->SetBinContent(i + 1, j + 1, acc[i][j]);
    }
  }

  auto c = new TCanvas("c", "Bigram heatmap", 900, 800);
  c->SetRightMargin(0.15);

  h->Draw("COLZ TEXT");

  c->Update();
  c->SaveAs("bigrams.png");
}

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

  torch::Tensor N = torch::zeros({ 27, 27 }, torch::kInt64);
  for (std::string word : lines) {
    word = "." + word + ".";

    for (size_t i = 0; i < word.size() - 1; i++) {
      char ch1 = word[i];
      char ch2 = word[i+1];

      auto ctoi = [](char ch) {
        if (ch == '.') return 26;
        return ch - 'a';
      };

      N.index_put_({ ctoi(ch1), ctoi(ch2) }, N.index({ ctoi(ch1), ctoi(ch2) }).add_(1));
    }
  }

  auto gen = at::make_generator<at::CPUGeneratorImpl>(2147483647);

  for (int i = 0; i < 50; i++) {
    int ix = 0;
    std::string name = "";
    for( ;; ) {
      auto probability = N.index({torch::indexing::Slice(), ix}).to(torch::kFloat);
      const float sum = N.index({torch::indexing::Slice(), ix}).to(torch::kFloat).sum().item<float>();
      ix = torch::multinomial(probability, 1, true, gen).item<int64_t>();
      if (ix == 26) { break; }
      name += static_cast<char>('a' + ix);
    }

    std::cout << name << std::endl;
  }
}