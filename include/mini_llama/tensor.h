#pragma once

#include <vector>
#include <string>


namespace mini_llama{
	struct Tensor{
		std::vector<float> data;
		std::vector<int> shape;

	//构造函数
	Tensor() = default;
	explicit Tensor(const std::vector<int>& input_shape, float fill = 0.0f);

	//num_dims, size, num_elements 这里全部完成,以防止报错
	size_t num_dims() const{return static_cast<int>(shape.size()); }

	size_t size() const{return data.size();}

	size_t num_elements() const {return data.size();}

  size_t FlatIndex(const std::vector<int>& indices) const;

  float& At(const std::vector<int>& indices);
  float At(const std::vector<int>& indices) const;

  //通过读数组的方式读写一维张量
  float& operator[](size_t i){return data[i]; }
  float operator[](size_t i) const {return data[i]; }

  std::string ShapeStringShort() const;
	
  //At1, At2, At3, At4 convenience accessors
  float& At1(int i);
  float At1(int i) const;

  float& At2(int i, int j);
  float At2(int i, int j) const;

  float& At3(int i, int j, int k);
  float At3(int i, int j, int k) const;

  float& At4(int i, int j, int k, int l);
  float At4(int i, int j, int k, int l) const;

  //RowPtr 快速获取2DTensor 的某一行的首地址
  float* RowPtr(int row);
  const float* RowPtr(int row) const;

  //Assert shape
  void AssertShape(const std::vector<int>& expected, const char* caller) const;

  //Reshape 
  Tensor ReshapeChecked(const std::vector<int>& new_shape, const char* caller) const; 
  
  std::string ShapeString() const { return ShapeStringShort(); }

  void Print(const std::string& name = " ", bool print_data = false) const;
  };// Tensor struct
    
  Tensor MakeTensor1D(int d0, float fill = 0.0f);

  Tensor MakeTensor2D(int d0, int d1, float fill = 0.0f);
 
  Tensor MakeTensor3D(int d0, int d1, int d2, float fill = 0.0f);
 
  Tensor MakeTensor4D(int d0, int d1, int d2, int d3, float fill = 0.0f);
  

} // namespace mini_llama
