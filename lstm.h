/* lstm.h
 */

//#ifndef NNET_LSTM_H_
//#define NNET_LSTM_H_

#include <ap_fixed.h>

#include <complex>
#include "ap_int.h"
#include "ap_fixed.h"

#include "nnet_lstm.h"

void lstm(
		FIX_16 lstm_in[N_TS*N1_LX],
		FIX_16 lstm1_out[N1_LX][N1_LH],
		FIX_16 GC_matrix[N1_LX][N1_LX]
);

//#endif /* _LSTM_H_ */
