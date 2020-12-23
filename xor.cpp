// xor.cpp
#include<Eigen/Dense>
#include<iostream>

Eigen::Vector2d hiddenLayers(int input,Eigen::MatrixXd Data, Eigen::MatrixXd W, 
                               Eigen::Vector2d c){
 // We use the Relu activation function
 Eigen::Vector2d inputs(Data(input,0), Data(input,1));
 Eigen::Vector2d w1(W(0,0),W(1,0));
 Eigen::Vector2d w2(W(0,1),W(1,1)); 

 // Calculation of activation values starts here
 double h1,h2;
 if(inputs.dot(w1) + c(0) < 0){
  h1 = 0;
 } else {
  h1 = inputs.dot(w1) + c(0);
 }

 if(inputs.dot(w2) + c(1) < 0){
  h2 = 0;
 } else {
  h2 = inputs.dot(w2) + c(1);
 }
 
 Eigen::Vector2d HiddenVals(h1,h2);
 return HiddenVals;
}

double outputLayer (Eigen::Vector2d h, Eigen::Vector2d weight, double bias){
 // We use linear regression  
 double val = weight.dot(h) + bias;
 return val;
}

double Xor(int input, Eigen::MatrixXd Data, Eigen::MatrixXd W, Eigen::Vector2d c,
              Eigen::Vector2d w, double b){
 Eigen::Vector2d h = hiddenLayers(input, Data, W, c);
 double output = outputLayer(h, w, b);
 return output;
}

int main(){
 /* Initialise the data matrix */
 Eigen::MatrixXd Data(4,2);
 Data << 0,0,
         0,1,
         1,0,
         1,1;

 /* Initialise the Weight matrix for the hidden layer */ 
 Eigen::MatrixXd Weight(2,2);
 Weight << 1,1,
           1,1;
 
 /* Initialise the C bias vector */
 Eigen::Vector2d C(0,-1);

 /* Initialise the Weight vector for output layer */
 Eigen::Vector2d w(1,-2);

 /* Initialise the b bias scalar */
 double bias = 0;
 
 /* Testing Functions */
 /*Eigen::Vector2d Layer1 = hiddenLayers(0,Data,Weight,C); 
 Eigen::Vector2d Layer2 = hiddenLayers(1,Data,Weight,C); 
 Eigen::Vector2d Layer3 = hiddenLayers(2,Data,Weight,C); 
 Eigen::Vector2d Layer4 = hiddenLayers(3,Data,Weight,C); 
 */

 std::cout << Xor(0,Data,Weight,C,w,bias) << std::endl;
 std::cout << Xor(1,Data,Weight,C,w,bias) << std::endl;
 std::cout << Xor(2,Data,Weight,C,w,bias) << std::endl;
 std::cout << Xor(3,Data,Weight,C,w,bias) << std::endl;
}

