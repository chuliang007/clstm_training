#ifndef NNET_LSTM_H_
#define NNET_LSTM_H_

//#include "nnet_common.h"
#include <cstdlib>
//#include "nnet_activation.h"
//#include "nnet_dense.h"
#include <hls_math.h>
#include <assert.h>
//#include "parameters.h"

#define N_TS 140
#define N1_LX 14
#define N1_LH 10

typedef ap_fixed<16, 8> FIX_16;
typedef ap_uint<1> uint1;

void hard_tanh(FIX_16 data[N1_LH], FIX_16 res[N1_LH])
{
	#pragma inline
	#pragma HLS PIPELINE

#pragma HLS ARRAY_PARTITION variable=res complete

    FIX_16 datareg;
    //FIX_16 slope = (FIX_16) 0.2;
    //FIX_16 shift = (FIX_16) 0.5;
    for (int ii=0; ii<N1_LH; ii++) {
#pragma HLS PIPELINE
        datareg = data[ii];
        if (datareg > 1)       datareg = (FIX_16)  1;
        else if (datareg < -1) datareg = (FIX_16) -1;
        //else                   ;
        res[ii] = datareg;
    }
}

void hard_sigmoid(FIX_16 data[N1_LH], FIX_16 res[N1_LH])
{
	#pragma inline
	#pragma HLS PIPELINE

#pragma HLS ARRAY_PARTITION variable=res complete

    FIX_16 datareg;
    FIX_16 slope = (FIX_16) 0.2;
    FIX_16 shift = (FIX_16) 0.5;
    for (int ii=0; ii<N1_LH; ii++) {
#pragma HLS PIPELINE
        datareg = slope * data[ii] + shift;
        if (datareg > 1) datareg = (FIX_16) 1;
        else if (datareg < 0) datareg = 0;
        res[ii] = datareg;
    }
}

void dense_simple_ix(
	FIX_16	data[N1_LX],
	FIX_16	res[N1_LH*4],
	FIX_16	weights[N1_LX*N1_LH*4],
	FIX_16	biases[N1_LH*4]
)
{
	// partition the weight array for pipeline
#pragma HLS ARRAY_PARTITION variable=weights complete
#pragma HLS ARRAY_PARTITION variable=data complete

    FIX_16 mult[N1_LX*N1_LH*4];
    FIX_16 acc[N1_LH*4];

    #pragma HLS ARRAY_PARTITION variable=mult complete
    #pragma HLS ARRAY_PARTITION variable=acc complete
	#pragma HLS ARRAY_PARTITION variable=res complete

	for(int ii = 0; ii < N1_LX; ii++) {
//#pragma HLS PIPELINE
		for(int jj = 0; jj < N1_LH*4; jj++) {
#pragma HLS PIPELINE
			int index = ii*N1_LH*4+jj;
			mult[index] = data[ii] * weights[index];
		}
	}

	for(int iacc = 0; iacc < N1_LH*4; iacc++) {
#pragma HLS PIPELINE
		acc[iacc] = (FIX_16) biases[iacc];
	}

	for(int ii = 0; ii < N1_LX; ii++) {
//#pragma HLS PIPELINE
		for(int jj = 0; jj < N1_LH*4; jj++) {
#pragma HLS PIPELINE
			int index = ii*N1_LH*4+jj;
			acc[jj] += mult[index];
		}
	}

	for(int ires = 0; ires < N1_LH*4; ires++){
#pragma HLS PIPELINE
		res[ires] = (FIX_16) (acc[ires]);
	}
}

void dense_simple_hx(
	FIX_16	data[N1_LH],
	FIX_16	res[N1_LH*4],
	FIX_16	weights[N1_LH*N1_LH*4],
	FIX_16	biases[N1_LH*4]
)
{
	// partition the weight array for pipeline
#pragma HLS ARRAY_PARTITION variable=weights complete
#pragma HLS ARRAY_PARTITION variable=data complete

    FIX_16 mult[N1_LH*N1_LH*4];
    FIX_16 acc[N1_LH*4];

    #pragma HLS ARRAY_PARTITION variable=mult complete
    #pragma HLS ARRAY_PARTITION variable=acc complete
	#pragma HLS ARRAY_PARTITION variable=res complete

	for(int ii = 0; ii < N1_LH; ii++) {
//#pragma HLS PIPELINE
		for(int jj = 0; jj < N1_LH*4; jj++) {
#pragma HLS PIPELINE
			int index = ii*N1_LH*4+jj;
			mult[index] = data[ii] * weights[index];
		}
	}

	for(int iacc = 0; iacc < N1_LH*4; iacc++) {
#pragma HLS PIPELINE
		acc[iacc] = (FIX_16) biases[iacc];
	}

	for(int ii = 0; ii < N1_LH; ii++) {
//#pragma HLS PIPELINE
		for(int jj = 0; jj < N1_LH*4; jj++) {
#pragma HLS PIPELINE
			int index = ii*N1_LH*4+jj;
			acc[jj] += mult[index];
		}
	}

	for(int ires = 0; ires < N1_LH*4; ires++){
//#pragma HLS PIPELINE
		res[ires] = (FIX_16) (acc[ires]);
	}
}

void dense_simple_bp(
	FIX_16	data[N1_LH],
	FIX_16	res[N1_LH*4],
	FIX_16	weights[N1_LH*N1_LH*4]
)
{
	// partition the weight array for pipeline
#pragma HLS ARRAY_PARTITION variable=weights complete
#pragma HLS ARRAY_PARTITION variable=data complete

    FIX_16 mult[N1_LH*N1_LH*4];
    FIX_16 acc[N1_LH*4];

    #pragma HLS ARRAY_PARTITION variable=mult complete
    #pragma HLS ARRAY_PARTITION variable=acc complete
	#pragma HLS ARRAY_PARTITION variable=res complete
	#pragma HLS ARRAY_PARTITION variable=mult complete

	for(int ires = 0; ires < N1_LH*4; ires++){
#pragma HLS PIPELINE
		acc[ires] = res[ires];
	}

	for(int ii = 0; ii < N1_LH; ii++) {
//#pragma HLS PIPELINE
		for(int jj = 0; jj < N1_LH*4; jj++) {
#pragma HLS PIPELINE
			int index = ii*N1_LH*4+jj;
			mult[index] = acc[jj] * weights[index];
		}
	}

	for(int ii = 0; ii < N1_LH; ii++) {
//#pragma HLS PIPELINE
		for(int jj = 0; jj < N1_LH*4; jj++) {
#pragma HLS PIPELINE
			int index = ii*N1_LH*4+jj;
			data[ii] += mult[index];
		}
	}
}

void weight_grad_calc(
    FIX_16	data[N1_LX],
    FIX_16	gate_f_error[N1_LH],
    FIX_16	gate_i_error[N1_LH],
    FIX_16	gate_g_error[N1_LH],
    FIX_16	gate_o_error[N1_LH],

	FIX_16	c_pre[N1_LH],
    FIX_16	weights_x[N1_LX*N1_LH*4],
	FIX_16	weights_h[N1_LH * N1_LH*4],
    FIX_16	biases[N1_LH*4]
)
{
	const FIX_16 lr = 0.01;
	// partition the weight array for pipeline
	const int weight_factor = N1_LH*4;
#pragma HLS ARRAY_PARTITION variable=weights_x complete //cyclic factor=weight_factor
#pragma HLS ARRAY_PARTITION variable=weights_h complete //cyclic factor=weight_factor
#pragma HLS ARRAY_PARTITION variable=biases complete //cyclic factor=weight_factor

    FIX_16 weights_x_tmp[N1_LX*N1_LH*4];
	FIX_16 weights_h_tmp[N1_LH * N1_LH*4];

#pragma HLS DEPENDENCE variable=weights_x inter false
#pragma HLS DEPENDENCE variable=weights_h inter false
#pragma HLS DEPENDENCE variable=biases inter false

#pragma HLS ARRAY_PARTITION variable=weights_x_tmp complete //cyclic factor=weight_factor
#pragma HLS ARRAY_PARTITION variable=weights_h_tmp complete //cyclic factor=weight_factor

    int index;
	FIX_16 c_pre_acc[N1_LH];
    FIX_16 acc[N1_LX];
    FIX_16 gate_error[N1_LH*4];
#pragma HLS ARRAY_PARTITION variable=acc complete //cyclic factor=weight_factor
#pragma HLS ARRAY_PARTITION variable=c_pre_acc complete //cyclic factor=weight_factor
#pragma HLS ARRAY_PARTITION variable=gate_error complete

    #pragma HLS ARRAY_PARTITION variable=acc complete
	#pragma HLS ARRAY_PARTITION variable=c_pre complete

    // input load
    for(int jj = 0; jj < N1_LX; jj++) {
#pragma HLS PIPELINE
    	acc[jj] = data[jj];
    }
    for(int jj = 0; jj < N1_LH; jj++) {
#pragma HLS PIPELINE
    	c_pre_acc[jj] = c_pre[jj];
    }

    // gate_error concat
	for(int jj = 0; jj < N1_LH; jj++) {
#pragma HLS PIPELINE
		gate_error[0*N1_LH + jj] = gate_f_error[jj];
		gate_error[1*N1_LH + jj] = gate_i_error[jj];
		gate_error[2*N1_LH + jj] = gate_g_error[jj];
		gate_error[3*N1_LH + jj] = gate_o_error[jj];
	}

    for(int jj = 0; jj < N1_LH*4; jj++) {
//#pragma HLS PIPELINE
    	// weight_h_grad
    	for(int ii = 0; ii < N1_LH; ii++) {
#pragma HLS PIPELINE
            int index = ii*N1_LH*4 + jj;
			weights_h[index] -= lr * c_pre_acc[ii] * gate_error[jj];
		}
    	// weight_x_grad
    	for(int ii = 0; ii < N1_LX; ii++) {
#pragma HLS PIPELINE
            int index = ii*N1_LH*4 + jj;
            weights_x[index] -= lr * acc[ii] * gate_error[jj];
		}
    	// bias updatess
    	biases[jj] -= lr * gate_error[jj];
    }
}

void lstm_tail(
	FIX_16 gate_f[N1_LH],
	FIX_16 gate_i[N1_LH],
	FIX_16 gate_g[N1_LH],
	FIX_16 gate_o[N1_LH],
//	FIX_16 h_pre[N1_LH],
    FIX_16 c_pre[N1_LH],
// output
    FIX_16 c_cur[N1_LH],
	FIX_16 h_cur[N1_LH]
){

    FIX_16 c_tmp1[N1_LH];
    FIX_16 c_tmp2[N1_LH];
    FIX_16 c_cur_activ[N1_LH];

//#pragma HLS PIPELINE II=CONFIG_T::reuse_factor_tail

	#pragma HLS ARRAY_PARTITION variable=c_pre complete
	#pragma HLS ARRAY_PARTITION variable=c_cur complete
	#pragma HLS ARRAY_PARTITION variable=h_cur complete

    #pragma HLS ARRAY_PARTITION variable=c_tmp1 complete
    #pragma HLS ARRAY_PARTITION variable=c_tmp2 complete
    #pragma HLS ARRAY_PARTITION variable=c_cur_activ complete

//    int multiplier_limit  = ceil( (3*float(N1_LH)) / float(CONFIG_T::reuse_factor_tail));
//    #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation

	CELL:
	for(int icell = 0; icell < N1_LH; icell++){
#pragma HLS PIPELINE
        c_tmp1[icell] = gate_f[icell] * c_pre[icell];
        c_tmp2[icell] = gate_i[icell] * gate_g[icell];
        c_cur[icell]  = c_tmp1[icell] + c_tmp2[icell];
	}

	hard_tanh (c_cur, c_cur_activ);  // tanh

	HIDDEN_UNITS:
	for(int itail = 0; itail < N1_LH; itail++){
#pragma HLS PIPELINE
        h_cur[itail] = gate_o[itail] * c_cur_activ[itail];
	}

};

void lstm_tail_backward(
	FIX_16 gate_f_error[N1_LH],
	FIX_16 gate_i_error[N1_LH],
	FIX_16 gate_g_error[N1_LH],
	FIX_16 gate_o_error[N1_LH],

    FIX_16 gate_f_activ[N1_LH],	//activation function output in the forward
	FIX_16 gate_f_t1_activ[N1_LH],
	FIX_16 gate_i_activ[N1_LH],
	FIX_16 gate_g_activ[N1_LH],
	FIX_16 gate_o_activ[N1_LH],

    FIX_16 c_pre[N1_LH],
    FIX_16 c_cur[N1_LH],
    FIX_16 error_c[N1_LH],
	FIX_16 error_c_t1[N1_LH],
	FIX_16 error_h[N1_LH],
	FIX_16 error_concat[N1_LH*4]
){

    FIX_16 c_tmp1[N1_LH];
    FIX_16 c_tmp2[N1_LH];
    FIX_16 c_cur_activ[N1_LH];

    #pragma HLS ARRAY_PARTITION variable=c_tmp1 complete
    #pragma HLS ARRAY_PARTITION variable=c_tmp2 complete
    #pragma HLS ARRAY_PARTITION variable=c_cur_activ complete

//#pragma HLS PIPELINE

    // tanh(c_cur)
    hard_tanh (c_cur, c_cur_activ);

    Error_Calc:
    for(int icell = 0; icell < N1_LH; icell++) {
#pragma HLS PIPELINE
    	error_c[icell] = error_h[icell] * gate_o_activ[icell] * (1 - c_cur[icell]*c_cur[icell]) + error_c_t1[icell] * gate_f_t1_activ[icell]; // omit partial error
    }

    Error_gates:
    for(int icell = 0; icell < N1_LH; icell++) {
#pragma HLS PIPELINE
        // gate_f
        gate_f_error[icell] = error_c[icell] * c_pre[icell] * gate_f_activ[icell] * (1 - gate_f_activ[icell]);
        // gate_i
        gate_i_error[icell] = error_c[icell] * gate_g_activ[icell] * (1 - gate_i_activ[icell]);
        // gate_g
        gate_g_error[icell] = error_c[icell] * gate_i_activ[icell] * (1 - gate_g_activ[icell]*gate_g_activ[icell]);
        // gate_o
        gate_o_error[icell] = error_h[icell] * c_cur_activ[icell] * gate_o_activ[icell] * (1 - gate_o_activ[icell]);
    }

    Error_concat:
    for(int icell = 0; icell < N1_LH; icell++) {
#pragma HLS PIPELINE
    	error_concat[0*N1_LH + icell] = gate_f_error[icell];
    	error_concat[1*N1_LH + icell] = gate_i_error[icell];
    	error_concat[2*N1_LH + icell] = gate_g_error[icell];
    	error_concat[3*N1_LH + icell] = gate_o_error[icell];
    }
};

void prox_update(
	uint1 GC_matrix[N1_LX],
	FIX_16 weights_x[N1_LX * N1_LH * 4]
)
{
	FIX_16 norm[N1_LX];
#pragma HLS ARRAY_PARTITION variable=norm complete

	const int weight_factor = N1_LH * 4;
#pragma HLS ARRAY_PARTITION variable=weights_x complete //cyclic factor=weight_factor dim=2

#pragma HLS DEPENDENCE variable=weights_x intra false
//#pragma HLS DEPENDENCE variable=weights_x inter false

	FIX_16 sum[N1_LX] = {0};
	#pragma HLS ARRAY_PARTITION variable=sum complete

	SUM:
	for (int j = 0; j < N1_LH * 4; j++) {
#pragma HLS PIPELINE
		for (int i = 0; i < N1_LX; i++) {
//	#pragma HLS PIPELINE
			int index = i * N1_LH * 4 + j;
			sum[i] += weights_x[index] * weights_x[index];
		}
	}
	NORM:
	for (int i = 0; i < N1_LX; i++) {
#pragma HLS PIPELINE
		norm[i] = (FIX_16) hls::sqrt(sum[i]);
		GC_matrix[i] = (norm[i] > 0) ? (uint1) 1 : (uint1) 0;
	}
	PROX:
	for (int j = 0; j < N1_LH * 4; j++) {
#pragma HLS PIPELINE
		for (int i = 0; i < N1_LX; i++) {
//	#pragma HLS PIPELINE
			int index = i * N1_LH * 4 + j;
			weights_x[index] *= (1 - norm[i]);
		}
		for (int i = 0; i < N1_LX; i++) {
//	#pragma HLS PIPELINE
			int index = i * N1_LH * 4 + j;
			if (weights_x[index] < 0) {
				weights_x[index] = 0;
			}
		}
	}
}

// LSTM layer without setting the sequence return 
// output: only the final hidden units 
/*
The input of the LSTM is always is a 3D array. (batch_size, time_steps, seq_len).
The output of the LSTM could be a 2D array or 3D array depending upon the return_sequences argument.
If return_sequence is False, the output is a 2D array. (batch_size, units)
If return_sequence is True, the output is a 3D array. batch_size, time_steps, units)
*/
void clstm_fw(
	//int index,
    FIX_16 data[N1_LX*N_TS],

    FIX_16 gate_f_activ[N1_LX][N_TS+1][N1_LH],
    FIX_16 gate_i_activ[N1_LX][N_TS][N1_LH],
    FIX_16 gate_g_activ[N1_LX][N_TS][N1_LH],
    FIX_16 gate_o_activ[N1_LX][N_TS][N1_LH],

    FIX_16 weights_x[N1_LX][N1_LX*N1_LH*4],
    FIX_16 weights_h[N1_LX][N1_LH*N1_LH*4],
	FIX_16 biases[N1_LX][N1_LH * 4],

	FIX_16  res[N1_LX][N1_LH]
){

	FIX_16 input_x[N1_LX];

    // array partition for res
	#pragma HLS ARRAY_PARTITION variable=res complete

    FIX_16 acc_x[N1_LH * 4];
    FIX_16 acc[N1_LH * 4];

    FIX_16 h_pre[N1_LH];
    FIX_16 h_cur[N1_LH];
    FIX_16 c_pre[N1_LH];
    FIX_16 c_cur[N1_LH];

    FIX_16 gate_f[N1_LH];
    FIX_16 gate_i[N1_LH];
    FIX_16 gate_g[N1_LH];
    FIX_16 gate_o[N1_LH];

    BUF_INT_FW:
    for(int ii = 0; ii < N1_LH; ii++){
#pragma HLS PIPELINE
        h_pre[ii] = 0;
        c_pre[ii] = 0;
    }

    INUTT_FW:
	for(int its = 0; its < N_TS; its++) {
//#pragma HLS PIPELINE

    	INUTT_FW_PIPELINE:
        for(int ix = 0; ix < N1_LX; ix++) {
	#pragma HLS PIPELINE

    		input_x[ix] = data[ix + its*N1_LX];

            dense_simple_ix(input_x, acc_x, weights_x[ix], biases[ix]);
            dense_simple_hx(h_pre, acc, weights_h[ix], acc_x);

			GATES_SPLIT:
            for(int igate = 0; igate < N1_LH; igate++) {
		 #pragma HLS UNROLL
				gate_f[igate] = acc[igate];
				gate_i[igate] = acc[1*N1_LH+igate];
				gate_g[igate] = acc[2*N1_LH+igate];
				gate_o[igate] = acc[3*N1_LH+igate];
			}

			//template<class FIX_16, class FIX_16, typename CONFIG_T>
            hard_sigmoid (gate_f, gate_f_activ[ix][its]);
			hard_sigmoid (gate_i, gate_i_activ[ix][its]);
			hard_tanh	 (gate_g, gate_g_activ[ix][its]);   // tanh
			hard_sigmoid (gate_o, gate_o_activ[ix][its]);

			lstm_tail(gate_f_activ[ix][its], gate_i_activ[ix][its], gate_g_activ[ix][its], gate_o_activ[ix][its], c_pre, c_cur, h_cur);

			BUF_UPD_FW:
            for(int ii = 0; ii < N1_LH; ii++) {
		 #pragma HLS UNROLL
				h_pre[ii] = h_cur[ii];
				c_pre[ii] = c_cur[ii];
            }

        	OUTPUT_FW:
        	for(int ii = 0; ii < N1_LH; ii++) {
        #pragma HLS PIPELINE
        		res[ix][ii] = h_cur[ii];
        	}
        }
	}

}// lstm_fw

void clstm_bw(
	//int index,
    FIX_16 data[N1_LX*N_TS],

    FIX_16 gate_f_activ[N1_LX][N_TS+1][N1_LH],
    FIX_16 gate_i_activ[N1_LX][N_TS][N1_LH],
    FIX_16 gate_g_activ[N1_LX][N_TS][N1_LH],
    FIX_16 gate_o_activ[N1_LX][N_TS][N1_LH],

    FIX_16 weights_x[N1_LX][N1_LX*N1_LH*4],
    FIX_16 weights_h[N1_LX][N1_LH*N1_LH*4],
	FIX_16 biases[N1_LX][N1_LH * 4],

	FIX_16 res[N1_LX][N1_LH]
){

	FIX_16 input_x[N1_LX];
	FIX_16 error_c[N1_LH];
	FIX_16 error_c_t1[N1_LH];
	FIX_16 error_h[N1_LH];
	FIX_16 error_concat[N1_LH*4];

	#pragma HLS ARRAY_PARTITION variable=input_x complete
	#pragma HLS ARRAY_PARTITION variable=error_c complete
	#pragma HLS ARRAY_PARTITION variable=error_h complete
	#pragma HLS ARRAY_PARTITION variable=error_concat complete

    FIX_16 h_pre[N1_LH];
    FIX_16 h_cur[N1_LH];
    FIX_16 c_pre[N1_LH];
    FIX_16 c_cur[N1_LH];

	FIX_16 gate_f_error[N1_LH];
    FIX_16 gate_i_error[N1_LH];
    FIX_16 gate_g_error[N1_LH];
    FIX_16 gate_o_error[N1_LH];

    BUF_INT_BW:
    for(int ii = 0; ii < N1_LH; ii++){
#pragma HLS PIPELINE
        h_pre[ii] = 0;
        c_pre[ii] = 0;
        error_c_t1[ii] = 0;
    }

	MSE_LOSS:
	for(int ii = 0; ii < N1_LH; ii++) {
		for(int ix = 0; ix < N1_LX; ix++) {
#pragma HLS PIPELINE
		    error_h[ii] += (res[ix][ii] - data[ii+1]) / N1_LX;	// error_h at timestep N_TS
		}
	}

    LSTM_BW:
	for(int its = N_TS - 1; its >= 0; its--) {
//#pragma HLS PIPELINE

        TAIL_WEIGHT_UPDATE:
        for(int ix = 0; ix < N1_LX; ix++) {
	#pragma HLS PIPELINE

        	input_x[ix] = data[ix + its*N1_LX];

            /* error back-propagation */
            lstm_tail_backward(gate_f_error, gate_i_error, gate_g_error, gate_o_error,
                               gate_f_activ[ix][its], gate_f_activ[ix][its+1], gate_i_activ[ix][its], gate_g_activ[ix][its], gate_o_activ[ix][its],
							   c_pre, c_cur, error_c, error_c_t1, error_h, error_concat);

            dense_simple_bp(error_h, error_concat, weights_h[ix]);	// omit weight transpose

            /* gradient calc and weight update */
            weight_grad_calc(input_x, gate_f_error, gate_i_error, gate_g_error, gate_o_error,
            				 c_pre, weights_x[ix], weights_h[ix], biases[ix]);

            OUTPUT_BW:
        	for(int ii = 0; ii < N1_LH; ii++){
         #pragma HLS UNROLL
				h_cur[ii] = h_pre[ii];
				c_cur[ii] = c_pre[ii];
        	}
        }
	}

}// lstm_bw

#endif
