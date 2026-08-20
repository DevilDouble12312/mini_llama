#include "mini_llama/tensor.h"

#include <iostream>
#include <iomanip>
#include <limits>
#include <string>
#include <stdexcept>
#include <vector>

namespace mini_llama{
namespace {

  const char* CallerName(const char* caller) {
    return caller == nullptr ? "Tensor" : caller;
  }

  std::string ShapeToString(const std::vector<int>& shape) {
    std::string s = "[";
    for(size_t i = 0; i < shape.size(); i++) {
      if(i > 0) {
        s += ", ";
      }
      s += std::to_string(shape[i]);
    }
    s += "]";
    return s;
  }

  size_t CheckedNumel(const std::vector<int>&shape, const char* caller){
    size_t total = 1;
    //判断两个,一维度小于0,抛出异常
    for(size_t axis = 0;axis < shape.size(); axis++ ) {
      const int dim = shape[axis];
      if (dim <= 0) {
      throw std::runtime_error(std::string(CallerName(caller)) +
                               ": dimension at axis " + std::to_string(axis) +
                               " must be positive, got " + std::to_string(dim) +
                               " in shape " + ShapeToString(shape));
    }

      //如果shape的总乘积溢出size_t上限,同样抛出异常
      const size_t dim_size = static_cast<size_t>(dim);
      if (total > std::numeric_limits<size_t>::max() / dim_size) {
        throw std::runtime_error(std::string(CallerName(caller)) +
                               ": shape element count overflow for" +
                               ShapeToString(shape));
    }
  total *= dim_size;    
  } 
  return total;
  }

  //checkAxisIndex
  //判定某一个轴有没有越过该张量shape的界限
  void CheckAxisIndex(const Tensor& t, int axis, int flat_index, const char* caller){
    const int dim = t.shape[axis];//获取某一维度具体值
    if (flat_index < 0 || flat_index >= dim) { //这里注意 dim是计数的维度,而flat_index是轴下标
      throw std::out_of_range(
          std::string(caller) + ":FlatIndex" + std::to_string(flat_index) + 
          " out of range for axis" + std::to_string(axis) + "with size" +
          std::to_string(dim) + "in tensor" + t.ShapeStringShort());
    }
  }

  void CheckRank(const Tensor& t, int expected, const char* caller){
    if (t.num_dims() != expected) {
      throw std::runtime_error(std::string(caller) + " expected" + 
          std::to_string(expected) + "D tensor, got" + 
          t.ShapeStringShort());
    }
  }


} //namespace 匿名命名空间


	Tensor::Tensor(const std::vector<int>& input_shape, float fill) : shape(input_shape){
		const size_t total = CheckedNumel(shape, "Tensor constructor");
    data.resize(total, fill);
	}

  //二维转一维
  size_t Tensor::FlatIndex(const std::vector<int>& indices) const {
   if(indices.size() != shape.size()) {
    throw std::runtime_error("Tensor::FlatIndex expected" +
        std::to_string(shape.size()) +
        " indices for tensor" + ShapeStringShort() + ", got" +
        std::to_string(indices.size()));
   } 
   for(size_t axis = 0; axis < indices.size(); ++axis) {
      CheckAxisIndex(*this, static_cast<int>(axis),indices[axis], "Tensor::FlatIndex" );
   }
   size_t flat = 0;
   size_t stride = 1;
   for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    flat += static_cast<size_t>(indices[i]) * stride;
    stride *= static_cast<size_t>(shape[i]);
   }
   return flat;
  }

  float& Tensor::At(const std::vector<int>& indices) {
    return data[FlatIndex(indices)];
  } 

  float Tensor::At(const std::vector<int>& indices) const {
    return data[FlatIndex(indices)];
  }

  std::string Tensor::ShapeStringShort() const { return ShapeToString(shape); }


  // At1, At2, At3, At4 Convenience accessors
  float& Tensor::At1(int i) {
    CheckRank(*this, 1, "At1");
    CheckAxisIndex(*this, 0, i, "At1");
    return data[i];
  }
  float Tensor::At1(int i) const {
    CheckRank(*this, 1, "At1");
    CheckAxisIndex(*this, 0, i, "At1");
    return data[i];
  }
  
  float& Tensor::At2(int i, int j){
    CheckRank(*this, 2, "At2");
    CheckAxisIndex(*this, 0, i, "At2");
    CheckAxisIndex(*this, 1, j, "At2");
    return data[j + shape[1]* i];
  }
  float Tensor::At2(int i, int j) const{
    CheckRank(*this, 2, "At2");
    CheckAxisIndex(*this, 0, i, "At2");
    CheckAxisIndex(*this, 1, j, "At2");
    return data[j + i * shape[1]];
  }


  float& Tensor::At3(int i, int j, int k){
    CheckRank(*this, 3, "At3");
    CheckAxisIndex(*this, 0, i, "At3");
    CheckAxisIndex(*this, 1, j, "At3");
    CheckAxisIndex(*this, 2, k, "At3");
    return data[k + j * shape[2] + i * shape[2] * shape[1]];
  }
  float Tensor::At3(int i, int j, int k) const{
    CheckRank(*this, 3, "At3");
    CheckAxisIndex(*this, 0, i, "At3");
    CheckAxisIndex(*this, 1, j, "At3");
    CheckAxisIndex(*this, 2, k, "At3");
    return data[k + j * shape[2] + i * shape[2] * shape[1]];
  }

  float& Tensor::At4(int i, int j, int k, int l){
    CheckRank(*this, 4, "At4");
    CheckAxisIndex(*this, 0, i, "At4");
    CheckAxisIndex(*this, 1, j, "At4");
    CheckAxisIndex(*this, 2, k, "At4");
    CheckAxisIndex(*this, 3, l, "At4");
    return data[l + k * shape[3] + j * shape[3] * shape[2] + i * shape[3] * shape[2] * shape[1]];
  }
  float Tensor::At4(int i, int j, int k, int l) const {
    CheckRank(*this, 4, "At4");
    CheckAxisIndex(*this, 0, i, "At4");
    CheckAxisIndex(*this, 1, j, "At4");
    CheckAxisIndex(*this, 2, k, "At4");
    CheckAxisIndex(*this, 3, l, "At4");
    return data[l + k * shape[3] + j * shape[3] * shape[2] + i * shape[3] * shape[2] * shape[1]];
  }



  //语法包装,更甜
  Tensor MakeTensor1D(int d0, float fill) {
    return Tensor ({d0}, fill);
  }
  
  Tensor MakeTensor2D(int d0, int d1, float fill){
    return Tensor ({d0, d1}, fill);
  }
 
  Tensor MakeTensor3D(int d0, int d1, int d2, float fill){
    return Tensor ({d0, d1, d2}, fill);
  }
 
  Tensor MakeTensor4D(int d0, int d1, int d2, int d3, float fill){
    return Tensor ({d0, d1, d2, d3}, fill);
  }

  float* Tensor::RowPtr(int row) {
    CheckRank(*this, 2, "row_ptr");
    CheckAxisIndex(*this, 0, row, "row_ptr");
    return data.data() + row * shape[1];
  }

  const float* Tensor::RowPtr(int row) const {
    CheckRank(*this, 2, "row_ptr");
    CheckAxisIndex(*this, 0, row, "row_ptr");
    return data.data() + row * shape[1];
  }

  void Tensor::AssertShape(const std::vector<int>& expected, const char* caller) const {
    if(shape != expected) {
      throw std::runtime_error(
          std::string(CallerName(caller)) + ":shape mismatch.expected " +
          ShapeToString(expected) + ", got" + ShapeStringShort());
    }
  }
  //Reshape Checked
  Tensor Tensor::ReshapeChecked(const std::vector<int>& new_shape, const char* caller) const {
    const size_t new_total = CheckedNumel(new_shape, CallerName(caller));
    if (new_total != data.size()) {//重塑允许改变结构,但是data总数一定不变
      throw std::runtime_error(
        std::string(CallerName(caller)) + ": cannot reshape tensor with " +
        std::to_string(data.size()) + " elements into shape with " +
        std::to_string(new_total) + " elements");
    }
    //const 函数承诺不修改原this,因此得创建副本t
    Tensor r = *this;
    r.shape = new_shape;
    return r;
  }

void Tensor::Print(const std::string& name, bool print_data) const {
  if (!name.empty()) {
    std::cout << name << " ";
  }
  std::cout << "shape=" << ShapeStringShort() << " size=" << size() << std::endl;
  if (print_data) {
    for (size_t i = 0; i < data.size(); ++i) {
      std::cout << std::fixed << std::setprecision(6) << data[i];
      if (i + 1 < data.size()) {
        std::cout << " ";
      }
    }
    std::cout << std::endl;
  }
}

}//namespace mini_llama





























