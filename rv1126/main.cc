/****************************************************************************
*
*    Copyright (c) 2017 - 2018 by Rockchip Corp.  All rights reserved.
*
*    The material in this file is confidential and contains trade secrets
*    of Rockchip Corporation. This is proprietary information owned by
*    Rockchip Corporation. No part of this work may be disclosed,
*    reproduced, copied, transmitted, or used in any way for any purpose,
*    without the express written permission of Rockchip Corporation.
*
*****************************************************************************/

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include <sstream>
#include <iomanip> 
//#include <math.h>
//#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "rknn_api.h"

#include "yolov3_post_process.h"

#include"rapidjson/include/document.h"
#include"rapidjson/include/writer.h"
#include"rapidjson/include/stringbuffer.h"

using namespace std;
using namespace cv;

/*-------------------------------------------
        Macros and Variables
-------------------------------------------*/

const string class_name[] = {"DaoXianYiWu","DiaoChe","WaJueJi","QiTaShiGongJiXie","ChanChe","TaDiao","TuiTuJi","FanDouChe","ShuiNiBengChe","YanWu","ShanHuo"};

/*-------------------------------------------
                  Functions
-------------------------------------------*/

static void printRKNNTensor(rknn_tensor_attr *attr) {
    printf("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n", 
            attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2], attr->dims[1], attr->dims[0], 
            attr->n_elems, attr->size, 0, attr->type, attr->qnt_type, attr->fl, attr->zp, attr->scale);
}

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp = fopen(filename, "rb");
    if(fp == nullptr) {
        printf("fopen %s fail!\n", filename);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    unsigned char *model = (unsigned char*)malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if(model_len != fread(model, 1, model_len, fp)) {
        printf("fread %s fail!\n", filename);
        free(model);
        return NULL;
    }
    *model_size = model_len;
    if(fp) {
        fclose(fp);
    }
    return model;
}


/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char** argv)
{
    const int img_width = 608;
    const int img_height = 608;
    const int img_channels = 3;

    rknn_context ctx;
    int ret;
    int model_len = 0;
    unsigned char *model;

    const char *model_path = argv[1];
    const char *img_path = argv[2];

    // Load image
    cv::Mat orig_img = cv::imread(img_path, 1);
    cv::Mat real_img = cv::imread(img_path, 1);
    // BGR2RGB
    cv::cvtColor(orig_img, orig_img, cv::COLOR_BGR2RGB);
    cv::Mat img = orig_img.clone();
    if(!orig_img.data) {
        printf("cv::imread %s fail!\n", img_path);
        return -1;
    }
    if(orig_img.cols != img_width || orig_img.rows != img_height) {
        printf("resize %d %d to %d %d\n", orig_img.cols, orig_img.rows, img_width, img_height);
        cv::resize(orig_img, img, cv::Size(img_width, img_height), (0, 0), (0, 0), cv::INTER_LINEAR);
    }

    // Load RKNN Model
    model = load_model(model_path, &model_len);
    ret = rknn_init(&ctx, model, model_len, 0);
    if(ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // Get Model Input Output Info
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        printRKNNTensor(&(input_attrs[i]));
    }

    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        printRKNNTensor(&(output_attrs[i]));
    }

    // Set Input Data
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = img.cols*img.rows*img.channels();
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = img.data;

    ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
    if(ret < 0) {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }

    // Run
    printf("rknn_run\n");
    ret = rknn_run(ctx, nullptr);
    if(ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }

    // Get Output
    rknn_output outputs[3];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;
    outputs[1].want_float = 1;
    outputs[2].want_float = 1;
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
    if(ret < 0) {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        return -1;
    }

    float out_pos[500*4];
    float out_prop[500];
    int out_label[500];
    int obj_num = 0;

    // Post Process
    obj_num = yolov3_post_process((float *)(outputs[0].buf), (float *)(outputs[1].buf), (float *)(outputs[2].buf), out_pos, out_prop, out_label);
    // Release rknn_outputs
    rknn_outputs_release(ctx, 3, outputs);

    // Draw Objects
    for (int i = 0; i < obj_num; i++) {
        printf("%s @ (%f %f %f %f) %f\n",
                class_name[out_label[i]].c_str(),
                out_pos[4*i+0], out_pos[4*i+1], out_pos[4*i+2], out_pos[4*i+3],
                out_prop[i]);
        int x1 = out_pos[4*i+0] * orig_img.cols;
        int y1 = out_pos[4*i+1] * orig_img.rows;
        int x2 = out_pos[4*i+2] * orig_img.cols;
        int y2 = out_pos[4*i+3] * orig_img.rows;
	//printf("x1:%d, y1:%d, x2:%d, y2:%d\n",x1,y1,x2,y2);
        rectangle(real_img, Point(x1, y1), Point(x2, y2), Scalar(255, 0, 0, 255), 3);
        putText(real_img, class_name[out_label[i]].c_str(), Point(x1, y1 - 12), 1, 2, Scalar(0, 255, 0, 255));
    }
	
    //printf("jpg name : %s\n", img_path);
    char jpgname[35];
    sprintf(jpgname,"./result/%s",img_path);
	
    imwrite(jpgname, real_img);

    if(obj_num > 0){
       
	rapidjson::StringBuffer s;
    rapidjson::Writer<rapidjson::StringBuffer> writer(s);
	string code = "0";
    string message = "Success";        
    writer.StartObject();
    writer.Key("code");
    writer.String(code.c_str());
    writer.Key("message");
    writer.String(message.c_str());        
    writer.Key("Data");
	writer.StartObject();	
    writer.Key("data");
    writer.StartArray();
	
        for (int i = 0; i < obj_num; i++){
	    //printf("%s @ (%f %f %f %f) %f\n",
		//class_name[out_label[i]].c_str(),
		//out_pos[4*i+0], out_pos[4*i+1], out_pos[4*i+2], out_pos[4*i+3],
		//out_prop[i]);
	    int a = int(out_prop[i]*1000 + 0.5);              
        float confidence = float(a)/1000;      	    
	    printf("origin confidence: %f\n", out_prop[i]);
        printf("confidence: %f\n", confidence);
        if(out_prop[i] > 0){
	        int x1 = out_pos[4*i+0] * orig_img.cols;
	        int y1 = out_pos[4*i+1] * orig_img.rows;
	    	int x2 = out_pos[4*i+2] * orig_img.cols;
	    	int y2 = out_pos[4*i+3] * orig_img.rows;
            writer.StartObject();
            writer.Key("label");
            writer.String(class_name[out_label[i]].c_str());
	    	writer.Key("confidence");
            writer.Double(out_prop[i]);
            writer.Key("xmin");
            writer.Int(x1);
            writer.Key("xmax");
            writer.Int(x2);
            writer.Key("ymin");
            writer.Int(y1);
            writer.Key("ymax");
            writer.Int(y2); 
            writer.EndObject();   
        }
            	       		
    }
    writer.EndArray();
	writer.EndObject();
    writer.EndObject();
	
    int imglen = strlen(img_path);
	//printf("imglen:%d\n",imglen);
    char imgname[imglen - 3];
	memset(imgname,'\0', sizeof(imgname));
    for( int j = 0; j < imglen-4; j++){
        imgname[j] = img_path[j];
    }
        
    //printf("jpgname:%s\n",img_path);
	//printf("imgname:%s\n",imgname);
	char jsonname[35];
    	sprintf(jsonname,"./result/%s.json",imgname);
	ofstream ofs; 
	ofs.open(jsonname);
	if (!ofs.is_open()){
	    return false; 
	}else{
	    ofs << s.GetString();
	    ofs.close();
	}	          	
    }else{
	rapidjson::StringBuffer s;
    rapidjson::Writer<rapidjson::StringBuffer> writer(s);
	string code = "-1";
    string message = "no object";        
    writer.StartObject();
    writer.Key("code");
    writer.String(code.c_str());
    writer.Key("message");
    writer.String(message.c_str());
	writer.EndObject();
	
    int imglen = strlen(img_path);	 
    char imgname[imglen - 3];
	memset(imgname,'\0', sizeof(imgname));
    for( int j = 0; j < imglen-4; j++){
        imgname[j] = img_path[j];
    }        
         
	char jsonname[35];
    sprintf(jsonname,"./result/%s.json",imgname);
	ofstream ofs; 
	ofs.open(jsonname);
	if (!ofs.is_open()){
	    return false; 
	}else{
	    ofs << s.GetString();
	    ofs.close();
	}	    
    }  
    	
    // Release
    if(ctx >= 0) {
        rknn_destroy(ctx);
    }
    if(model) {
        free(model);
    }
    return 0;
}
