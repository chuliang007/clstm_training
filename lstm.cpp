#include "math.h"
#include "lstm.h"

//#include "parameters.h"
#include "nnet_lstm.h"

#define NNET_LSTM_H_

#include "lstm1_wx.h"
#include "lstm1_wh.h"
#include "lstm1_wb.h"

//FIX_16 weights_x[N1_LX][N1_LX*N1_LH*4];
//FIX_16 weights_h[N1_LX][N1_LH*N1_LH*4];
//FIX_16 biases[N1_LX][N1_LH*4];

static FIX_16 gate_f_activ[N1_LX][N_TS+1][N1_LH] = {0};
static FIX_16 gate_i_activ[N1_LX][N_TS][N1_LH] = {0};
static FIX_16 gate_g_activ[N1_LX][N_TS][N1_LH] = {0};
static FIX_16 gate_o_activ[N1_LX][N_TS][N1_LH] = {0};

//#pragma hls_design top
void lstm(
		FIX_16 lstm_in[N_TS*N1_LX],
		FIX_16 lstm1_out[N1_LX][N1_LH],
		uint1  GC_matrix[N1_LX][N1_LX]
)
{
//#pragma HLS INTERFACE m_axi depth=720 port=lstm_in offset=slave bundle=lstm_in			// 20 * 36 = 720
//#pragma HLS INTERFACE m_axi depth=14400 port=lstm1_out offset=slave bundle=lstm1_out		// 36 * 20 * 20 = 14400
//#pragma HLS INTERFACE m_axi depth=1296 port=GC_matrix offset=slave bundle=GC_matrix		// 36 * 36 = 1296
//#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    #pragma HLS ARRAY_PARTITION variable=lstm_in complete //cyclic factor=input_factor
	#pragma HLS ARRAY_PARTITION variable=GC_matrix complete dim=2

	// lstm forward
	clstm_fw(lstm_in, gate_f_activ, gate_i_activ, gate_g_activ, gate_o_activ,  weights_x, weights_h, biases,  lstm1_out);
	// lstm backward
	clstm_bw(lstm_in, gate_f_activ, gate_i_activ, gate_g_activ, gate_o_activ,  weights_x, weights_h, biases,  lstm1_out);
	// prox weight update and NGC matrix out

	NGC_OUT:
	for(int ix = 0; ix < N1_LX; ix++) {
#pragma HLS PIPELINE
		prox_update(GC_matrix[ix], weights_x[ix]);
	}
}
