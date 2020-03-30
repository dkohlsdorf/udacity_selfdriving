#include "tools.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth)
{
   assert(estimations.size() > 0);
   assert(estimations.size() == ground_truth.size());
   assert(estimations[0].size() == ground_truth[0].size());

   int d = estimations[0].size();
   VectorXd rmse(d);
   for (int i = 0; i < d; i++)
   {
      rmse(i) = 0;
   }

   for (int i = 0; i < estimations.size(); i++)
   {
      VectorXd err = ground_truth[i] - estimations[i];
      err = err.array() * err.array();
      rmse += err;
   }
   rmse /= estimations.size();
   rmse = rmse.array().sqrt();
   return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd &x_state)
{
   float position_x = x_state(0);
   float position_y = x_state(1);
   float velocity_x = x_state(2);
   float velocity_y = x_state(3);

   float scaler1 = position_x * position_x + position_y * position_y;
   float scaler2 = sqrt(scaler1);
   float scaler3 = scaler1 * scaler2;

   MatrixXd Hj(3, 4);
   if (fabs(scaler1) < 0.0001)
   {
      std::cout << "CalculateJacobian () - Error - Division by Zero" << std::endl;
      return Hj;
   }
   float h00 = (position_x / scaler2);
   float h01 = (position_y / scaler2);
   float h10 = -(position_y / scaler1);
   float h11 = (position_x / scaler1);
   float h20 = position_y * (velocity_x * position_y - velocity_y * position_x) / scaler3;
   float h21 = position_x * (position_x * velocity_y - position_y * velocity_x) / scaler3;
   float h22 = position_x / scaler2;
   float h23 = position_y / scaler2;
   Hj << h00, h00, 0, 0,
       h10, h11, 0, 0,
       h20, h21, h22, h23;
   return Hj;
}
