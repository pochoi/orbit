data {
  int<lower=0> NUM_OF_OBS;    // number of time points of training 
//  int<lower=0> Npred;   // number of time points of prediction
  int<lower=0> NUM_OF_REGRESSOR;    // number of covariates
  matrix[NUM_OF_OBS,  NUM_OF_REGRESSOR] REGRESSOR_MAT;   // training covariate matrix
//  matrix[Npred, NUM_OF_REGRESSOR] Xpred;  // prediction covariate matrix
  vector[NUM_OF_OBS] RESPONSE;       // response
}
parameters {
  real alpha;           // constant mean
  vector[NUM_OF_REGRESSOR] beta;       // coefficients for covariates
  real<lower=0> obs_sigma;  // error scale
}
model {
  RESPONSE ~ normal(alpha + REGRESSOR_MAT * beta, obs_sigma);  // likelihood
}
//generated quantities {
//  vector[N] y_fit = REG_MAT * beta;
//  vector[Npred] y_pred = Xpred * beta;
//  real vsigma[Npred] = rep_array(obs_sigma, Npred);
//  vector[Npred] y_pred_sample = multi_normal_rng(Xpred * beta, diag_matrix(to_vector(vsigma)));
//}