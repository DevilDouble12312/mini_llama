#include <gtest/gtest.h>

#include "mini_llama/tensor.h"
#include <iostream>

using namespace mini_llama;

TEST(TensorTest, TestTensorShapeAndSize) {
	Tensor t1({3, 4, 5}, 0.0f);
	EXPECT_EQ(t1.num_dims(), 3);
	EXPECT_EQ(t1.size(), 60);
	EXPECT_EQ(t1.num_elements(), 60);
	EXPECT_EQ(t1.shape[0], 3);
	EXPECT_EQ(t1.shape[1], 4);
	EXPECT_EQ(t1.shape[2], 5);
}


TEST(TensorTest, TestTensorFill) {
  Tensor t({10}, 3.14f);
  EXPECT_EQ(t.size(), 10);
  for(size_t i= 0; i < t.size(); i++) {
    EXPECT_NEAR(t.data[i], 3.14f, 1e-6f);
  }
}


TEST(TensorTest, TestTensorRejectsZeroDimension) {
  try {
    Tensor t({2, 0}, 0.0f);
    FAIL() << "expected exception for zero tensor dimension!";
  } catch (const std::runtime_error& e){
    std::cout << "✅ Caught expected exception: " << e.what() << std::endl;
    // expected
  }
}


TEST(TensorTest, TestTensorRejectsNegativeDimension) {
  try {
    Tensor t({2, -3}, 0.0f);
    FAIL() << "expected execetion for negative tensor dimension!";
  } catch (const std::runtime_error& e) {
    std::cout << "✅ Caught expected exception: " << e.what() << std::endl;
    //expected
  }
}


TEST(Tensor, TestTensorIndexing) {
  Tensor t({2, 3}, 0.0f);
  t.At({0, 0}) = 1.0f;
  t.At({0, 1}) = 2.0f;
  t.At({1, 2}) = 6.0f;
  EXPECT_NEAR(t.At({0, 0}), 1.0f, 1e-6f);
  EXPECT_NEAR(t.At({0, 1}), 2.0f, 1e-6f);
  EXPECT_NEAR(t.At({1, 2}), 6.0f, 1e-6f);
}


TEST(TensorTest, TestTensorIndexWrongRank) {
  Tensor t({2, 3}, 0.0f);
  try {
    t.At({1});
    FAIL() << "expected exception for wrong FlatIndex rank";
  } catch (const std::runtime_error& e) {
    std::cout << "✅ Caught expected exception:" << e.what() << std::endl;
  }
}


TEST(TensorTest, TestTensorIndexNegative) {
  Tensor t({2, 3}, 0.0f);
  ASSERT_THROW(t.FlatIndex({-1,0}), std::out_of_range) << "错误, 测试数据超出shape界限,应该抛出out_of_range异常";
  std::cout << "✅ Test passed: Out-of-range was correctly rejected!" << std::endl;
}


TEST(TensorTest, TestTensorTooLarge) {
  Tensor t({2, 3}, 0.0f);
  ASSERT_THROW(t.FlatIndex({0, 3}), std::out_of_range) << "错误, 测试数据超出shape界限,应该抛出out_of_range异常";
  
  std::cout << "✅ Test passed: Out-of-range was correctly rejected!" << std::endl; 
}

TEST(TensorTest, TestTensor1dAccess) {
  Tensor t({5}, 0.0f);
  t[2] = 42.0f;
  EXPECT_NEAR(t[2], 42.0f, 1e-6f);
}

TEST(TensorTest, TestMakeTensorHelpers) {
  auto t1 = MakeTensor1D(10, 1.0f);
  EXPECT_EQ(t1.num_dims(), 1);
  EXPECT_EQ(t1.shape[0], 10);

  auto t2 = MakeTensor2D(3, 4, 1.0f);
  EXPECT_EQ(t2.num_dims(), 2);
  EXPECT_EQ(t2.shape[0], 3);
  EXPECT_EQ(t2.shape[1], 4);

  auto t3 = MakeTensor3D(2, 3, 4, 1.0f);
  EXPECT_EQ(t3.num_dims(), 3);
  EXPECT_EQ(t3.size(), 24);

  auto t4 = MakeTensor4D(2, 3, 4, 5, 1.0f);
  EXPECT_EQ(t4.num_dims(), 4);
  EXPECT_EQ(t4.size(), 120);

}
// At1, At2, At3, At4 convenience accessors

TEST(TensorTest, TestTensorAt1) {
  Tensor t({5}, 0.0f);
  t.At1(2) = 7.0f;
  EXPECT_NEAR(t.At1(2), 7.0f, 1e-6f);
}

TEST(TensorTest, TestTensorAt2) {
  Tensor t({2, 3}, 0.0f);
  t.At2(0, 1) = 5.0f;
  t.At2(1, 2) = 9.0f;
  EXPECT_NEAR(t.At2(0, 1), 5.0f, 1e-6f);
  EXPECT_NEAR(t.At2(1, 2), 9.0f, 1e-6f);
}

TEST(TensorTest, TestTensorAt2OutOfRange) {
  Tensor t({2, 3}, 0.0f);
  ASSERT_THROW(t.At2(2,0), std::out_of_range) << "expected exception for At2 row of range";
  std::cout << "✅ Test passed: Out-of-range was correctly rejected!" << std::endl;
}


TEST(TensorTest, TestTensorAt3) {
  Tensor t({2, 2, 2}, 0.0f);
  t.At3(1, 0, 1) = 3.0f;
  EXPECT_NEAR(t.At3(1, 0, 1), 3.0f, 1e-6f);
}

TEST(TensorTest, TestTensorAt4) {
  Tensor t({2, 2, 2, 2}, 0.0f);
  t.At4(1, 1, 0, 0) = 4.0f;
  EXPECT_NEAR(t.At4(1, 1, 0, 0), 4.0f, 1e-6f);
}

// row_ptr test
TEST(TensorTest, TestTensorRowPtr) {
  Tensor t({3, 4}, 0.0f);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      t.At2(i, j) = static_cast<float>(i * 10 + j);
    }
  }
  const float* row1 = t.RowPtr(1);
  EXPECT_NEAR(row1[0], 10.0f, 1e-6f);
  EXPECT_NEAR(row1[3], 13.0f, 1e-6f);
}

TEST(TensorTest, TestTensorRowPtrWrongDim) {
  Tensor t({5}, 0.0f);
  ASSERT_THROW(t.RowPtr(0), std::runtime_error) << "expected exception for rowptr runtime_error";
  std::cout << "✅ Test passed: Out-of-range was correctly rejected!" << std::endl;
}


TEST(TensorTest, TestTensorRowPtrOutOfRange) {
  Tensor t({3, 4}, 0.0f);
  ASSERT_THROW(t.RowPtr(3), std::out_of_range) << "expected exception for rowptr out_of_range";
  std::cout << "✅ Test passed: Out-of-range was correctly rejected!" << std::endl;
}

//AssertShape
TEST(TensorTest, TestTensorAssertShapeFail) {
  Tensor t({2, 3, 4}, 0.0f);
  t.AssertShape({2, 3, 4}, "TestPass");
}

TEST(TenosrTest, TestTensorAssertShapeFail) {
  Tensor t({2,3,4}, 0.0f);
  ASSERT_THROW(t.AssertShape({2,3,5}, "TestFail"), std::runtime_error) << "expected exception for AssertSHape runtime_error";
  std::cout << "✅ Test passed: Out-of-range was correctly rejected!" << std::endl;
}


//ReShape 

TEST(TensorTest, TestTensorReshapeCheckedPass) {
  Tensor t({2, 3}, 0.0f);
  for (int i = 0; i < 6; ++i) {
    t[i] = static_cast<float>(i);
  }
  Tensor r = t.ReshapeChecked({3, 2}, "TestReshape");
  EXPECT_EQ(r.num_dims(), 2);
  EXPECT_EQ(r.shape[0], 3);
  EXPECT_EQ(r.shape[1], 2);
  EXPECT_NEAR(r.At2(0, 0), 0.0f, 1e-6f);
  EXPECT_NEAR(r.At2(2, 1), 5.0f, 1e-6f);
}

TEST(TensorTest, TestTensorReshapeCheckedFail) {
  Tensor t({2, 3}, 0.0f);
  ASSERT_THROW(t.ReshapeChecked({4, 2}, "TestReshapeFail"), std::runtime_error) << "expected expetion for incompation reshape runtime_error";
  std::cout << "✅ Test passed: runtime_error was correctly rejected!" << std::endl;
}

TEST(TensorTest, TestTensorReshapeCheckedNegativeDim) {
  Tensor t({2, 3}, 0.0f);
  ASSERT_THROW(t.ReshapeChecked({-1, 6}, "TestReshapeNeg"), std::runtime_error) << "expected expetion for negative dimesion runtime_e      rror";
  std::cout << "✅ Test passed: runtime_error was correctly rejected!" << std::endl;
}


// ShapeString alias
TEST(TensorTest, TestTensorShapeStringAlias) {
  Tensor t({2, 3, 4}, 0.0f);
  EXPECT_TRUE(t.ShapeStringShort() == t.ShapeString());
}

//print
TEST(TensorTest, TestTensorPrint) {
    Tensor t({2, 3}, 1.0f);
    for (int i = 0; i < 6; ++i) t[i] = static_cast<float>(i * 0.5f);
    
    EXPECT_NO_THROW(t.Print("test", true));
}












