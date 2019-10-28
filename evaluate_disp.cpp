#include <iostream>
#include <stdio.h>
#include <math.h>
#include <opencv2/core.hpp>

#include "io_disp.h"
#include "io_integer.h"
#include "utils.h"

#define NUM_TEST_IMAGES 200
#define NUM_ERROR_IMAGES 20
#define ABS_THRESH 3.0
#define REL_THRESH 0.05

using namespace std;
using namespace cv;

vector<float> disparityErrorsOutlier (DisparityImage &D_gt,DisparityImage &D_orig,DisparityImage &D_ipol,IntegerImage &O_map) {

  // check file size
  if (D_gt.width()!=D_orig.width() || D_gt.height()!=D_orig.height()) {
    cout << "ERROR: Wrong file size!" << endl;
    throw 1;
  }

  // extract width and height
  int32_t width  = D_gt.width();
  int32_t height = D_gt.height();

  // init errors
  vector<float> errors;
  int32_t num_errors_bg = 0;
  int32_t num_pixels_bg = 0;
  int32_t num_errors_bg_result = 0;
  int32_t num_pixels_bg_result = 0;
  int32_t num_errors_fg = 0;
  int32_t num_pixels_fg = 0;
  int32_t num_errors_fg_result = 0;
  int32_t num_pixels_fg_result = 0;
  int32_t num_errors_all = 0;
  int32_t num_pixels_all = 0;
  int32_t num_errors_all_result = 0;
  int32_t num_pixels_all_result = 0;

  // for all pixels do
  for (int32_t u=0; u<width; u++) {
    for (int32_t v=0; v<height; v++) {
      if (D_gt.isValid(u,v)) {
        float d_gt  = D_gt.getDisp(u,v);
        float d_est = D_ipol.getDisp(u,v);
        float d_orig = D_orig.getDisp(u,v);
        bool  d_err = fabs(d_gt-d_est)>ABS_THRESH && fabs(d_gt-d_est)/fabs(d_gt)>REL_THRESH;
        if (O_map.getValue(u,v)==0) {
          if (d_err)
            num_errors_bg++;
          num_pixels_bg++;
          if (D_orig.isValid(u,v)) {
            if (d_err)
              num_errors_bg_result++;
            num_pixels_bg_result++;
          }
        } else {
          if (d_err)
            num_errors_fg++;
          num_pixels_fg++;
          if (D_orig.isValid(u,v)) {
            if (d_err)
              num_errors_fg_result++;
            num_pixels_fg_result++;
          }
        }
        if (d_err)
          num_errors_all++;
        num_pixels_all++;
        if (D_orig.isValid(u,v)) {
          if (d_err)
            num_errors_all_result++;
          num_pixels_all_result++;
        }
      }
    }
  }

  // push back errors and pixel count
  errors.push_back(num_errors_bg);
  errors.push_back(num_pixels_bg);
  errors.push_back(num_errors_bg_result);
  errors.push_back(num_pixels_bg_result);
  errors.push_back(num_errors_fg);
  errors.push_back(num_pixels_fg);
  errors.push_back(num_errors_fg_result);
  errors.push_back(num_pixels_fg_result);
  errors.push_back(num_errors_all);
  errors.push_back(num_pixels_all);
  errors.push_back(num_errors_all_result);
  errors.push_back(num_pixels_all_result);

  // push back density
  errors.push_back((float)num_pixels_all_result/max((float)num_pixels_all,1.0f));

  // return errors
  return errors;
}

bool eval (string directory, string image_name) 
{
    // ground truth and result directories
    string gt_img_dir = directory + "/" + "image_2";
    string gt_obj_map_dir = directory + "/" + "obj_map";
    string gt_disp_noc_0_dir = directory + "/" + "disp_noc_0";
    string gt_disp_occ_0_dir = directory + "/" + "disp_occ_0";
    string gt_disp_noc_1_dir = directory + "/" + "disp_noc_1";
    string gt_disp_occ_1_dir = directory + "/" + "disp_occ_1";
    string result_dir = "results";
    string result_disp_0_dir = result_dir + "/data/disp_0";

    // create output directories
    system(("mkdir -p " + result_dir + "/image_0/").c_str());
    system(("mkdir -p " + result_dir + "/errors_disp_noc_0/").c_str());
    system(("mkdir -p " + result_dir + "/errors_disp_occ_0/").c_str());
    system(("mkdir -p " + result_dir + "/errors_disp_img_0/").c_str());
    system(("mkdir -p " + result_dir + "/result_disp_img_0/").c_str());

    // accumulators
    float errors_disp_noc_0[3*4]     = {0,0,0,0,0,0,0,0,0,0,0,0};
    float errors_disp_occ_0[3*4]     = {0,0,0,0,0,0,0,0,0,0,0,0};
    float errors_disp_noc_1[3*4]     = {0,0,0,0,0,0,0,0,0,0,0,0};
    float errors_disp_occ_1[3*4]     = {0,0,0,0,0,0,0,0,0,0,0,0};
    float errors_flow_noc[3*4]       = {0,0,0,0,0,0,0,0,0,0,0,0};
    float errors_flow_occ[3*4]       = {0,0,0,0,0,0,0,0,0,0,0,0};
    float errors_scene_flow_noc[3*4] = {0,0,0,0,0,0,0,0,0,0,0,0};
    float errors_scene_flow_occ[3*4] = {0,0,0,0,0,0,0,0,0,0,0,0};

    // declaration of global data structures
    DisparityImage D_gt_noc_0, D_gt_occ_0, D_orig_0, D_ipol_0;

    // load object map (0:background, >0:foreground)
    IntegerImage O_map = IntegerImage(gt_obj_map_dir + "/" + image_name);

    // copy left camera image 
    string img_src = gt_img_dir + "/" + image_name;
    string img_dst = result_dir + "/image_0/" + image_name;
    system(("cp " + img_src + " " + img_dst).c_str());

    ///////////////////////////////////////////////////////////////////////////////////////////
    // evaluation of disp
    
    // load ground truth disparity maps
    D_gt_noc_0 = DisparityImage(gt_disp_noc_0_dir + "/" + image_name);
    D_gt_occ_0 = DisparityImage(gt_disp_occ_0_dir + "/" + image_name);
    
    string image_file = result_disp_0_dir + "/" + image_name;
    // load submitted result and interpolate missing values
    D_orig_0 = DisparityImage(image_file);
    D_ipol_0 = DisparityImage(D_orig_0);
    D_ipol_0.interpolateBackground();

    // calculate disparity errors
    vector<float> errors_noc_curr = disparityErrorsOutlier(D_gt_noc_0,D_orig_0,D_ipol_0,O_map);
    vector<float> errors_occ_curr = disparityErrorsOutlier(D_gt_occ_0,D_orig_0,D_ipol_0,O_map);

    // accumulate errors
    for (int32_t j=0; j<errors_noc_curr.size()-1; j++) {
        errors_disp_noc_0[j] += errors_noc_curr[j];
        errors_disp_occ_0[j] += errors_occ_curr[j];
    }

    // save error images
//    if (i<NUM_ERROR_IMAGES) {
    if (1) {

        int str_len = image_name.length();
        string prefix = image_name.substr(0, str_len-4);

        // save errors of error images to text file
        FILE *errors_noc_file = fopen((result_dir + "/errors_disp_noc_0/" + prefix + ".txt").c_str(),"w");
        FILE *errors_occ_file = fopen((result_dir + "/errors_disp_occ_0/" + prefix + ".txt").c_str(),"w");
        for (int32_t i=0; i<12; i+=2)
            fprintf(errors_noc_file,"%f ",errors_noc_curr[i]/max(errors_noc_curr[i+1],1.0f));
        fprintf(errors_noc_file,"%f ",errors_noc_curr[12]);
        for (int32_t i=0; i<12; i+=2)
            fprintf(errors_occ_file,"%f ",errors_occ_curr[i]/max(errors_occ_curr[i+1],1.0f));
        fprintf(errors_occ_file,"%f ",errors_occ_curr[12]);
        fclose(errors_noc_file);
        fclose(errors_occ_file);

        // save error image
        D_ipol_0.errorImage(D_gt_noc_0,D_gt_occ_0,true).write(result_dir + "/errors_disp_img_0/" + prefix + ".png");

        // compute maximum disparity
        float max_disp = D_gt_occ_0.maxDisp();

        // save interpolated disparity image false color coded
        D_ipol_0.writeColor(result_dir + "/result_disp_img_0/" + prefix + ".png",max_disp);
    }

    string stats_file_name;
    FILE *stats_file;

    // write summary statistics for disparity evaluation
    stats_file_name = result_dir + "/stats_disp_noc_0.txt";
    stats_file = fopen(stats_file_name.c_str(),"w");
    for (int32_t i=0; i<12; i+=2)
        fprintf(stats_file,"%f ",errors_disp_noc_0[i]/max(errors_disp_noc_0[i+1],1.0f));
    fprintf(stats_file,"%f ",errors_disp_noc_0[11]/max(errors_disp_noc_0[9],1.0f));
    fprintf(stats_file,"\n");
    fclose(stats_file);
    stats_file_name = result_dir + "/stats_disp_occ_0.txt";
    stats_file = fopen(stats_file_name.c_str(),"w");
    for (int32_t i=0; i<12; i+=2)
        fprintf(stats_file,"%f ",errors_disp_occ_0[i]/max(errors_disp_occ_0[i+1],1.0f));
    fprintf(stats_file,"%f ",errors_disp_occ_0[11]/max(errors_disp_occ_0[9],1.0f));
    fprintf(stats_file,"\n");
    fclose(stats_file);

    // success
    return true;
}



