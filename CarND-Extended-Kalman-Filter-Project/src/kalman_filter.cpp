#include "kalman_filter.h"
#include <math.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

double normalize_angle(double angle) {
  while(angle < -M_PI) {
    angle += 2 * M_PI;
  }
  while(angle > M_PI) {
    angle -= 2 * M_PI;
  }
  return angle;
}

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in; 	// object state
  P_ = P_in;  // object covariance matrix
  F_ = F_in;  // state transition matrix
  H_ = H_in;  // measurement matrix
  R_ = R_in;  // measurement covariance matrix
  Q_ = Q_in;  // process covariance matrix
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd y  = z - H_ * x_;
  MatrixXd Ht = H_.transpose();
  MatrixXd S  = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K =  P_ * Ht * Si;
  x_ = x_ + (K * y);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  VectorXd h = VectorXd(3);
  h << 
    sqrt(x_(0) * x_(0) + x_(1) * x_(1)), 
    atan2(x_(1), x_(1)), 
    (x_(0) * x_(2) + x_(1) * x_(3)) / sqrt(x_(0) * x_(0) + x_(1) * x_(1));
  VectorXd y = z - h;
  y(1) = normalize_angle(y(1));
  MatrixXd Ht = H_.transpose();
  MatrixXd S  = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K =  P_ * Ht * Si;
  x_ = x_ + (K * y);
  P_ = (I - K * H_) * P_;
}

