#include <torch/torch.h>
#include <ATen/CPUGeneratorImpl.h>

#include <iostream>
#include <fstream>
#include <span>
#include <tuple>
#include <ranges>
#include <map>
#include <algorithm>

#include <TCanvas.h>
#include <TH2I.h>
#include <TStyle.h>

const int ALPHABET_SIZE = 27;
const float LEARNING_RATE = 0.01;

void draw_bigram_root(torch::Tensor N) {
  gStyle->SetOptStat(0);
  gStyle->SetPaintTextFormat("g"); 

  N = N.to(torch::kCPU).contiguous();

  auto h = new TH2I("h", "Bigram counts;second letter;first letter",
                    ALPHABET_SIZE, 0, ALPHABET_SIZE,
                    ALPHABET_SIZE, 0, ALPHABET_SIZE);

  for (int i = 0; i < ALPHABET_SIZE; ++i) {
    TString lab; lab += char('a' + i - 1);
    if (i == 0) { lab = '.'; }
    h->GetXaxis()->SetBinLabel(i + 1, lab);
    h->GetYaxis()->SetBinLabel(i + 1, lab);
  }

  auto acc = N.accessor<int64_t, 2>();
  for (int i = 0; i < ALPHABET_SIZE; ++i) {
    for (int j = 0; j < ALPHABET_SIZE; ++j) {
      h->SetBinContent(j + 1, i + 1, acc[i][j]);
    }
  }

  auto c = new TCanvas("c", "Bigram heatmap", 900, 800);
  c->SetRightMargin(0.15);

  h->Draw("COLZ TEXT");

  c->Update();
  c->SaveAs("bigrams.png");
}

auto ctoi = [](char ch) -> int64_t {
  if (ch == '.') return 0;
  // if (ch == '}') return 27;
  return (ch - 'a') + 1;
};

auto itoc = [](int ch) -> char {
  if (ch == 0) return '.';
  // if (ch == 27) return '{';
  return static_cast<char>('a' + ch - 1);
};

int bigram() {

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

  torch::Tensor N = torch::zeros({ ALPHABET_SIZE, ALPHABET_SIZE }, torch::kInt64);
  for (std::string word : lines) {
    word = "." + word + ".";

    for (size_t i = 0; i < word.size() - 1; i++) {
      char ch1 = word[i];
      char ch2 = word[i+1];

      N.index_put_({ ctoi(ch1), ctoi(ch2) }, N.index({ ctoi(ch1), ctoi(ch2) }).add_(1));
    }
  }

  auto gen = torch::make_generator<torch::CPUGeneratorImpl>(2147483647);

  // smoothing is applied to remove infinite loss cases where bigram has zero appearences
  const int SMOOTHING = 1; 
  auto probability = (N + SMOOTHING).to(torch::kFloat);
  probability = probability / probability.sum(1, true);

  for (int i = 0; i < 50; i++) {
    int ix = 0;
    std::string name = "";
    
    for( ;; ) {
      auto local_probability = probability.index({ix, torch::indexing::Slice()});

      ix = at::multinomial(local_probability, 1, true, gen).item<int64_t>();

      if (ix == 0) { break; }
      name += itoc(ix);
    }

    std::cout << name << std::endl;
  }

  // calculate likelihood
  torch::Tensor log_likelihood = torch::zeros({1});
  int n = 0;
  for (std::string word : (lines | std::views::take(1))) {
    word = "." + word+"qj" + ".";


    for (size_t i = 0; i < word.size() - 1; i++) {
      char ch1 = word[i];
      char ch2 = word[i+1];
      auto prob = probability.index({ ctoi(ch1), ctoi(ch2) });
      auto logprob = torch::log(prob);
      log_likelihood += logprob;
      n++; 

      float _prob = prob.item<float>();
      float _logprob = prob.item<float>();

      std::cout << ch1 << ch2 << ", prob: " << _prob << ", logprob: " << _logprob << std::endl;
    }

    std::cout << "log_likelihood: " << log_likelihood << std::endl;
    torch::Tensor negative_log_likelihood = -log_likelihood;
    std::cout << "negative_log_likelihood: " << negative_log_likelihood << ", avg_nll: " << negative_log_likelihood / n << std::endl; 
  }

  draw_bigram_root(N);

  return 0;
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

  std::vector<int64_t> xs_buffer, ys_buffer;

  // for (std::string word : lines | std::views::take(1)) {
  for (std::string word : lines) {
    word = "." + word + ".";

    for (size_t i = 0; i < word.size() - 1; i++) {
      char ch1 = word[i];
      char ch2 = word[i+1];
      printf("%c", ch1);
      xs_buffer.push_back(ctoi(ch1));
      ys_buffer.push_back(ctoi(ch2));
    }
  }

  torch::Tensor xs = torch::from_blob(
    xs_buffer.data(),
    {(long)xs_buffer.size()},
    torch::kInt64
  ).clone();

  torch::Tensor ys = torch::from_blob(
    ys_buffer.data(),
    {(long)ys_buffer.size()},
    torch::kInt64
  ).clone(); 
  
  // randomly initialize 27 neurons' weights, each neuron receives 27 inputs
  auto gen = torch::make_generator<torch::CPUGeneratorImpl>(2147483647);
  torch::Tensor W = torch::randn({ 27, 27}, gen);
  W.set_requires_grad(true);

  /* begin forward pass */
  // input to the network
  torch::Tensor xenc = torch::nn::functional::one_hot(xs, ALPHABET_SIZE).to(torch::kFloat);

  for (size_t i = 0; i < 10000; i++) {
    // predict log-counts
    torch::Tensor logits = torch::matmul(xenc, W);
    // extract counts
    torch::Tensor counts = logits.exp();
    // get probabilities for the next character
    torch::Tensor probs = counts / counts.sum(1, true);
    // last two lines together are called softmax: take outputs of a neural net layer and outputs probability distributions (normalization function), you can put this after any other linear operation
    
    torch::Tensor loss = -probs.index({torch::arange(5), ys}).log().mean();
    
    /* end forward pass */

    printf("loss: %f", loss.item<float>());
    /* begin backward pass */
    if (W.grad().defined()) {
        W.grad().zero_();
    }
    loss.backward();

    W.data() -= LEARNING_RATE * W.grad();
    /* end backward pass */

  }
  
}