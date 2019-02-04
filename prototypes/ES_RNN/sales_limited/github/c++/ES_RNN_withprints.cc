/*ES-RNN: ES-RNN Exponential Smoothing Recurrent Neural Network hybrid. Point forecast.
Slawek Smyl,  Jan-May 2017.

Dilated LSTMs, with optional shortcuts, attention.
It is meant to be used for Monthly and Quarterly series of M4 competition, becasue the DE (Diversified Ensemble) version is too slow.
The program uses and requires Dynet NN library(https://github.com/clab/dynet); can be compiled and run on Windows, Linux, and Mac.

It has to be invoked in pair of executables, passing at least two integers: seedForChunks, chunkNo
so e.g. create a script with following lines on Windows
start <this_executable> 10 1
start <this_executable> 10 2
Modern computers have at more then 2 cores, so e.g. on 6-core machine create and run the following script with 3 pairs of workers:
# start <this_executable> 10 1 0
# start <this_executable> 10 2 0
# start <this_executable> 20 1 5
# start <this_executable> 20 2 5
# start <this_executable> 30 1 10
# start <this_executable> 30 2 10
seedForChunks have to be the same withion one pair, chunk numbers have to be 1 and 2. 
We have added here the third parameter: ibigOffset. The straddle should be equal or bigger than BIG_LOOP.
Each pair goes through BIG_LOOP (by default 3, change in code below if you want) of model fitting and prediction, 
so 2 pairs, as above, will produce 6 forecasts to be ensembled later, in R.
By increasing number of pairs, e.g. to 6 on 12-core computer, one can reduce BIG_LOOP to 1, so reduce execution time, and still have 6 forecasts - 
a decent number to ensemble (in a separate R script).

There are three blocks of parameters below, one active (starting with //PARAMS--------------) and two inactive. 
The active block is setup as in the final run of forecasting quarterly series. Similarly Monthly block. 
The Daily block is more of a demo, allowing to run quickly forecast for Daily series, although with slightly worse performance (use another program ES_RNN_E.cc for it). It was not used for the final submission. 
So, you need comment/uncomment to have one block of interest active.


*/

//#define USE_ODBC
//define USE_ODBC if you want to 
// 1. run the program in backtesting mode (which means you also need to set LBACK>0 below. Read the comment below.
// 2. save forecasts to a datatabase. Mysql and SQL Server were tested. The table creation and some other scripts should be found in \sql directory of the source code.
// Of course setting up ODBC is not that simple, :-), e.g. you need to create DSN=slawek, that points to a database with the output table.
// Saving to the db is convenient, but not necessary - all forecasts are always saved to as csv files in automatically created subdirectory (sorry sometimes two directories, so you have to copy :-)) of OUTPUT_DIR
//If saving to database you need to modify run varaible, for each new run, otherwise you will get the table key error.

#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/expr.h"
#include "dynet/io.h"
#include "dynet/model.h"
#include "dynet/nodes.h"
#include "dynet/expr.h"
#include "dynet/lstm.h"
#include "slstm.h" //my implementation of dilated LSTMs

#if defined USE_ODBC        
  #if defined _WINDOWS
    #include <windows.h>
  #endif  
  #include <sqlext.h>
  #include <sql.h>
#endif 

#include <ctime>
#include <numeric>
#include <array> 
#include <fstream>
#include <algorithm>  
#include <math.h> 

using namespace std;
using namespace dynet;



// string DATA_DIR = "f:\\progs\\data\\M4DataSet\\"; //with the competition data csvs //windows file paths corresponding ot slawek's local machine
string DATA_DIR="/home/deegup/sales_data/Train/"; //use with linux
// string OUTPUT_DIR = "f:\\progs\\data\\M4\\"; 
string OUTPUT_DIR="/home/deegup/sales_data/output/";

int LBACK = 0; //LBACK 0 means final mode: learning on all data and forecasting. LBACK=1 would move back by OUTPUT_SIZE, and forecast last known OUTPUT_SIZE points, for backtesting. LBACK could be a larger integer, but then number of series shrinks.


//PARAMS--------------
string VARIABLE = "Quarterly";
const string run = "50/45 (1,2),(4,8), LR=0.001/{10,1e-4f}, EPOCHS=15, LVP=80 40*"; 
const float PERCENTILE = 50; //we always use Pinball loss, although on normalized values. When forecasting point value, we actually forecast median, so PERCENTILE=50
const float TRAINING_PERCENTILE = 45;  //the program has a tendency for positive bias. So, we can reduce it by running smaller TRAINING_PERCENTILE.

vector<vector<unsigned>> dilations={{1,2},{4,8}};//Each vector represents one chunk of Dilateed LSTMS, connected in standard resnNet fashion
//#define USE_RESIDUAL_LSTM
//#define USE_ATTENTIVE_LSTM  //so for Quarterly series, we do not use either the more advanced residual connections nor attention.
const bool ADD_NL_LAYER=false;  //whether to insert a tanh() layer between the RNN stack and the linear adaptor (output) layer

const float INITIAL_LEARNING_RATE = 0.001f;
const map<int, float> LEARNING_RATES = { { 10,1e-4f } }; //at which epoch we set them up to what
const float PER_SERIES_LR_MULTIP = 1; //multiplier for per-series parameters' learning rate.

const int NUM_OF_TRAIN_EPOCHS = 8;
const unsigned int STATE_HSIZE = 40;

const int SEASONALITY = 4;
const unsigned int INPUT_SIZE = 4;
const int INPUT_SIZE_I= INPUT_SIZE;
const unsigned int OUTPUT_SIZE = 3;
const int OUTPUT_SIZE_I= OUTPUT_SIZE;
const int MIN_INP_SEQ_LEN = 0;
const float LEVEL_VARIABILITY_PENALTY = 80;  //Multiplier for L" penalty against wigglines of level vector. Important.
const int MIN_SERIES_LENGTH = INPUT_SIZE_I + OUTPUT_SIZE_I+ MIN_INP_SEQ_LEN+2;
const int MAX_SERIES_LENGTH = 40 * SEASONALITY + MIN_SERIES_LENGTH; //we are chopping longer series, to last, max e.g. 40 years



/*
string VARIABLE = "Monthly";
const string run = "50/49 Res (1,3,6,12), LR=5e-4 {12,1e-4f}, EPOCHS=10, 20*";  
const float PERCENTILE = 50; //we always use Pinball loss, although on normalized values. When forecasting point value, we actually forecast median, so PERCENTILE=50
const float TRAINING_PERCENTILE = 49;  //the program has a tendency for positive bias. So, we can reduce it by running smaller TRAINING_PERCENTILE

#define USE_RESIDUAL_LSTM  //so for Monthly we use only one block, so no standard resNet shortcuts, but instead but of the special residual shortcuts, after https://arxiv.org/abs/1701.03360.
//#define USE_ATTENTIVE_LSTM
const bool ADD_NL_LAYER = false;  //whether to insert a tanh() layer between the RNN stack and the linear adaptor (output) layer

vector<vector<unsigned>> dilations={{1,3,6,12}};//so for Monthly we use only one block, so no standard resNet shortcut
const float INITIAL_LEARNING_RATE = 5e-4;
const map<int, float> LEARNING_RATES = { { 12,1e-4f } }; //at which epoch we set them up to what
const float PER_SERIES_LR_MULTIP = 1; //multiplier for per-series parameters' learning rate.

const int NUM_OF_TRAIN_EPOCHS = 10;
const unsigned int STATE_HSIZE = 50;

const float LEVEL_VARIABILITY_PENALTY = 50;  //Multiplier for L" penalty against wigglines of level vector.

const int SEASONALITY = 12;
const unsigned int OUTPUT_SIZE = 18;
const unsigned int INPUT_SIZE = 12;
const int INPUT_SIZE_I= INPUT_SIZE;
const int OUTPUT_SIZE_I= OUTPUT_SIZE;

const int MIN_INP_SEQ_LEN = 0;
const int MIN_SERIES_LENGTH = INPUT_SIZE_I + OUTPUT_SIZE_I+ MIN_INP_SEQ_LEN+2;
const int MAX_SERIES_LENGTH = 20 * SEASONALITY + MIN_SERIES_LENGTH; //we are chopping longer series, to last, max e.g. 40 years
*/


/*
string VARIABLE = "Daily";
const string run = "50/49  NL LRMult=1.5, 3/5 (1,7,28) LR=3e-4 {9,1e-4f} EPOCHS=15, LVP=100 HSIZE=40 20w";

//#define USE_RESIDUAL_LSTM
//#define USE_ATTENTIVE_LSTM
const bool ADD_NL_LAYER = true;

const int PERCENTILE = 50; //we always use Pinball loss. When forecasting point value, we actually forecast median, so PERCENTILE=50
const int TRAINING_PERCENTILE = 49;  //the program has a tendency for positive bias. So, we can reduce it by running smaller TRAINING_PERCENTILE

const int SEASONALITY_NUM = 1; //0 means no seasonality, for Yearly; 1 - single seasonality for Daily(7), Weekly(52); 2 - dual seaonality for Hourly (24,168)
const int SEASONALITY = 7;
const int SEASONALITY2 = 0;
vector<vector<unsigned>> dilations = { { 1,7,28 } };

const float INITIAL_LEARNING_RATE = 3e-4;
const map<int, float> LEARNING_RATES = { { 9,1e-4f } }; //at which epoch we manually set them up to what
const float PER_SERIES_LR_MULTIP = 1.5;
const int NUM_OF_TRAIN_EPOCHS = 15;

float LEVEL_VARIABILITY_PENALTY = 100;  //Multiplier for L" penalty against wigglines of level vector.
const unsigned int STATE_HSIZE = 40;

const unsigned int INPUT_SIZE = 7;
const int INPUT_SIZE_I = INPUT_SIZE;
const unsigned int OUTPUT_SIZE = 14;
const int OUTPUT_SIZE_I = OUTPUT_SIZE;

const int MIN_INP_SEQ_LEN = 0;
const int MIN_SERIES_LENGTH = OUTPUT_SIZE + INPUT_SIZE + MIN_INP_SEQ_LEN + 2;  //this is compared to n==(total length - OUTPUT_SIZE). Total length may be truncated by LBACK
const int MAX_SERIES_LENGTH = 20 * SEASONALITY + MIN_SERIES_LENGTH;  //we are chopping longer series, to max of last 20 years
*/

Expression squash(Expression& x) {
  return log(x);
}

Expression expand(Expression& x) {
  return exp(x);
}

string INPUT_PATH = DATA_DIR + "time_series_nonan.csv";
string INFO_INPUT_PATH = DATA_DIR + "sales-info.csv";

#if defined _DEBUG
  const int MAX_NUM_OF_SERIES = 40;
#else
  const int MAX_NUM_OF_SERIES = -1; //use all series
#endif // _DEBUG

const unsigned int NUM_OF_CATEGORIES = 11;//in data provided
const int BIG_LOOP = 3;
const int NUM_OF_CHUNKS = 2;
const float EPS=1e-6;
const int AVERAGING_LEVEL=5;
const bool USE_MEDIAN = false;
const int MIDDLE_POS_FOR_AVG = 2; //if using medians

const float NOISE_STD=0.001; 
const int FREQ_OF_TEST=1;
const float GRADIENT_CLIPPING=20;
const float C_STATE_PENALTY = 0;

const float BIG_FLOAT=1e38;//numeric_limits<float>::max();
const bool PRINT_DIAGN=true;
const float TAU = PERCENTILE / 100.;
const float TRAINING_TAU = TRAINING_PERCENTILE / 100.;
const unsigned ATTENTION_HSIZE=STATE_HSIZE;

const bool USE_AUTO_LEARNING_RATE=false;
//if USE_AUTO_LEARNING_RATE, and only if LBACK>0
const float MIN_LEARNING_RATE = 0.0001f;
const float LR_RATIO = sqrt(10);
const float LR_TOLERANCE_MULTIP = 1.005;
const int L3_PERIOD = 2;
const int MIN_EPOCHS_BEFORE_CHANGING_LRATE = 2;


#if defined USE_ODBC
  void HandleDiagnosticRecord(SQLHANDLE      hHandle,
    SQLSMALLINT    hType,
    RETCODE        RetCode);

  #if defined _WINDOWS
    WCHAR* pwszConnStr = L"DSN=slawek";
  #else
    SQLCHAR* pwszConnStr = (SQLCHAR*) "DSN=slawek";
  #endif   
  #define TRYODBC(h, ht, x)   {   RETCODE rc = x;\
                                if (rc != SQL_SUCCESS) \
                                { \
                                    HandleDiagnosticRecord (h, ht, rc); \
                                } \
                                if (rc == SQL_ERROR) \
                                { \
                                    fprintf(stderr, "Error in " #x "\n"); \
                                    if (hStmt)    { \
																			SQLFreeHandle(SQL_HANDLE_STMT, hStmt); \
																		} \
																		if (hDbc)    { \
																			SQLDisconnect(hDbc); \
																			SQLFreeHandle(SQL_HANDLE_DBC, hDbc); \
																		} \
																		if (hEnv)    { \
																				SQLFreeHandle(SQL_HANDLE_ENV, hEnv); \
																		} \
																		exit(-1); \
                                }  \
                            }

#endif

struct M4TS {//storing series data
  vector < float> categories_vect;
  vector<float> vals;
  vector<float> testVals;//empty, unless LBACK>0
  int n;
  
  M4TS(string category, stringstream  &line_stream) {
    array<float, NUM_OF_CATEGORIES> categories = { 0,0,0,0,0,0,0,0,0,0,0 }; //construct one-hot vector indicating which categories are used
    std::cout << "current category: " << category << endl;
    if (category == "1.0")
      categories[0] = 1;
    else if (category == "2.0")
      categories[1] = 1;
    else if (category == "3.0")
      categories[2] = 1;
    else if (category == "4.0")
      categories[3] = 1;
    else if (category == "5.0")
      categories[4] = 1;
    else if (category == "6.0")
      categories[5] = 1;
    else if (category == "7.0")
      categories[6] = 1;
    else if (category == "8.0")
      categories[7] = 1;
    else if (category == "9.0")
      categories[8] = 1;
    else if (category == "10.0")
      categories[9] = 1;
    else if (category == "11.0")
      categories[10] = 1;
    else {
      cerr << "unknown brand...";
      exit(-1);
    }
    for (int i = 0; i < NUM_OF_CATEGORIES; i++)
      categories_vect.push_back(categories[i]);

    string tmp_str;
    while(getline(line_stream, tmp_str, ',' )) { //takes in parameters from line stream (seedsForChunks, chunkNum)
      string val_str;
      for (const auto c : tmp_str) {
        if (c != '\"') {//remove quotes
          val_str.push_back(c);
        }
      }
      if (val_str.size() == 0)
        break;
      float val=(atof(val_str.c_str()));
      vals.push_back(val);
    }
    if (LBACK > 0) { //extract last OUTPUT_SIZE points as the test values
      if (vals.size() > LBACK*OUTPUT_SIZE_I) {
        auto first = vals.begin() + vals.size() - LBACK*OUTPUT_SIZE_I;
        auto pastLast = vals.begin() + vals.size() - (LBACK-1)*OUTPUT_SIZE_I;
        vector<float> input_vect(first, pastLast); //[first,pastLast)
        testVals= input_vect;
        vals.resize(vals.size() - LBACK*OUTPUT_SIZE_I); //remove last LBACK*OUTPUT_SIZE elements
        n = vals.size();
      } else
        n = 0;
    } else {
      n = vals.size();
    }
    if (n > MAX_SERIES_LENGTH) { //chop long series
      vals.erase(vals.begin(), vals.begin() + (n-MAX_SERIES_LENGTH)); //remove some early data
      n = vals.size();
    }
  }
  M4TS(){};
};


struct AdditionalParams {//Per series, important
  Parameter levSm;
  Parameter sSm;
  array<Parameter, SEASONALITY> initSeasonality;
};

struct AdditionalParamsF {//Used for storing diagnostics
  float levSm;
  float sSm;
  array<float, SEASONALITY> initSeasonality;
  vector<float> levels;
  vector<float> seasons;
};


Expression pinBallLoss(const Expression& out_ex, const Expression& actuals_ex) {//used by Dynet, learning loss function
  vector<Expression> losses;
  for (unsigned int indx = 0; indx<OUTPUT_SIZE; indx++) {
    auto forec = pick(out_ex, indx);
    auto actual = pick(actuals_ex, indx);
    if (as_scalar(actual.value()) > as_scalar(forec.value()))
      losses.push_back((actual - forec)*TRAINING_TAU);
    else
      losses.push_back((actual - forec)*(TRAINING_TAU - 1));
  }
  return sum(losses) / OUTPUT_SIZE * 2;
}


//weighted quantile Loss, used just for diagnostics, if if LBACK>0 and PERCENTILE!=50
float wQuantLoss(vector<float>& out_vect, vector<float>& actuals_vect) { //Looks like a swalek thing (not standard)
  float sumf = 0; float suma=0;
  for (unsigned int indx = 0; indx<OUTPUT_SIZE; indx++) {
    auto forec = out_vect[indx];
    auto actual = actuals_vect[indx];
    suma+= abs(actual);
    if (actual > forec)
      sumf = sumf + (actual - forec)*TAU;
    else
      sumf = sumf + (actual - forec)*(TAU - 1);
  }
  return sumf / suma * 200;
}

//used just for diagnostics, if LBACK>0 and PERCENTILE==50
float sMAPE(vector<float>& out_vect, vector<float>& actuals_vect) {
  float sumf = 0;
  for (unsigned int indx = 0; indx<OUTPUT_SIZE; indx++) {
    auto forec = out_vect[indx];
    auto actual = actuals_vect[indx];
    sumf+=abs(forec-actual)/(abs(forec)+abs(actual));
  }
  return sumf / OUTPUT_SIZE * 200;
}

float errorFunc(vector<float>& out_vect, vector<float>& actuals_vect) { //use sMAPE or wQuant loss
  if (PERCENTILE==50)
    return sMAPE(out_vect, actuals_vect);
  else
    return wQuantLoss(out_vect, actuals_vect);
}

int main(int argc, char** argv) {
  std::cout << "here" << endl;
  dynet::initialize(argc, argv);

  int seedForChunks = 10; //Yes it runs, without any params, but it will work only on 1/NUM_OF_CHUNKS of all cases. The system is expected to run in NUM_OF_CHUNKS multiples.
  int chunkNo = 1;
  int ibigOffset = 0;
  if (argc >= 3) {
    seedForChunks = atoi(argv[1]);
    chunkNo = atoi(argv[2]);
  } 
  if (argc >= 4)
	  ibigOffset = atoi(argv[3]);

  if (chunkNo > NUM_OF_CHUNKS) {
    cerr << "chunkNo > NUM_OF_CHUNKS";
    exit(-1);
  }
  else if (chunkNo <= 0) {
    cerr << "chunkNo <= 0";
    exit(-1);
  }

  cout<<VARIABLE<<" "<<run<<endl;
  std::cout << "seed:" << seedForChunks << " chunk no:" << chunkNo;
  if (ibigOffset>0) 
    std::cout<< " ibigOffset:"<< ibigOffset;  //if continuing prematurely stopped run
  if (LBACK>0) 
    std::cout<<" lback:"<<LBACK;
  std::cout<<endl;

   if  (USE_AUTO_LEARNING_RATE && LBACK == 0) {
    cerr<<"Can't use auto learning rate when LBACK==0";
    exit(-1);
   }

 
  time_t rawtime;
  struct tm * timeinfo;
  char buffer[80];

  time(&rawtime);
  timeinfo = localtime(&rawtime);

  strftime(buffer, sizeof(buffer), "%Y-%m-%d_%I_%M", timeinfo);
  std::string timestamp_str(buffer);

  ostringstream convert2;

  #if defined _WINDOWS
    OUTPUT_DIR = OUTPUT_DIR + "\\" + VARIABLE+ timestamp_str;
    if (LBACK==0) 
      OUTPUT_DIR = OUTPUT_DIR+"Final\\";
    OUTPUT_DIR = OUTPUT_DIR + convert2.str();
    string exec = string("mkdir ") + OUTPUT_DIR;  //so occasionaly, if the programs do not start within the same minute, you may find more than one output dir created. After the run just manullay put them together.
  #else
    OUTPUT_DIR = OUTPUT_DIR + "/" + VARIABLE + timestamp_str;
    if (LBACK == 0)
      OUTPUT_DIR = OUTPUT_DIR + "Final/";
    OUTPUT_DIR = OUTPUT_DIR + convert2.str();
    string exec = string("mkdir -p ") + OUTPUT_DIR;
  #endif
  system(exec.c_str());

  if (LBACK == 0) 
    cout << "Doing final of " << VARIABLE << " into " << OUTPUT_DIR << endl;

#if defined USE_ODBC
  time_t t = time(0);   // get time now
  struct tm * now = localtime(&t);
  TIMESTAMP_STRUCT now_ts;
  now_ts.year= now->tm_year+1900;
  now_ts.month=now->tm_mon+1;
  now_ts.day=now->tm_mday;
  now_ts.hour=now->tm_hour;
  now_ts.minute=now->tm_min;
  now_ts.second=now->tm_sec;
  now_ts.fraction=0; //reportedly needed

  const int OFFSET_TO_FIRST_ACTUAL=5;
  string insertQuery_str = "insert into M72nn(run, LBack, ibig, series, epoch ";
  for (int iq = 1; iq <= OUTPUT_SIZE_I; iq++) {
    stringstream ss;
    ss << iq;
    string iq_str = ss.str();
    insertQuery_str = insertQuery_str +", actual"+iq_str+", forec" + iq_str;
  }
  insertQuery_str = insertQuery_str +", trainingError, variable, n, dateTimeOfPrediction) \
    values(? , ? , ? , ? , ? ";
  for (int iq = 1; iq <= OUTPUT_SIZE_I; iq++) {
    insertQuery_str = insertQuery_str + ",?,?";
  }
  insertQuery_str = insertQuery_str + ",?,?,?,?)";
  #if defined _WINDOWS  
  wstring insertQuery(insertQuery_str.begin(), insertQuery_str.end());
  SQLWCHAR* sqlQuery = (SQLWCHAR*)insertQuery.c_str();
  #else
  SQLCHAR* sqlQuery =(SQLCHAR*)insertQuery_str.c_str();
  #endif


  SQLHENV  hEnv = NULL;
  SQLHDBC  hDbc = NULL;
  SQLHSTMT hStmt = NULL, hInsertStmt = NULL;

  if (SQLAllocHandle(SQL_HANDLE_ENV, SQL_NULL_HANDLE, &hEnv) == SQL_ERROR) {
    fprintf(stderr, "Unable to allocate an environment handle\n");
    exit(-1);
  }
  TRYODBC(hEnv,
    SQL_HANDLE_ENV,
    SQLSetEnvAttr(hEnv,
      SQL_ATTR_ODBC_VERSION,
      (SQLPOINTER)SQL_OV_ODBC3,
      0));

  // Allocate a connection
  TRYODBC(hEnv,
    SQL_HANDLE_ENV,
    SQLAllocHandle(SQL_HANDLE_DBC, hEnv, &hDbc));

  TRYODBC(hDbc,
    SQL_HANDLE_DBC,
    SQLDriverConnect(hDbc,
      NULL,
      pwszConnStr,
      SQL_NTS,
      NULL,
      0,
      NULL,
      SQL_DRIVER_COMPLETE));
  fprintf(stderr, "Connected!\n");

  TRYODBC(hDbc,
    SQL_HANDLE_DBC,
    SQLSetConnectAttr(hDbc, SQL_ATTR_AUTOCOMMIT, (SQLPOINTER)SQL_AUTOCOMMIT_OFF, SQL_IS_INTEGER));

  TRYODBC(hDbc,
    SQL_HANDLE_DBC,
    SQLAllocHandle(SQL_HANDLE_STMT, hDbc, &hInsertStmt));

  TRYODBC(hInsertStmt,
    SQL_HANDLE_STMT,
    SQLPrepare(hInsertStmt, sqlQuery, SQL_NTS));

  SQLLEN nullTerminatedStringOfRun = SQL_NTS;
  SQLLEN nullTerminatedStringOfSeries = SQL_NTS;
  SQLLEN nullTerminatedStringOfVariable = SQL_NTS;

  TRYODBC(hInsertStmt,
    SQL_HANDLE_STMT,
    SQLBindParameter(hInsertStmt, 1, SQL_PARAM_INPUT, SQL_C_CHAR, SQL_VARCHAR, 0, 0, (SQLCHAR*)run.c_str(), 0, &nullTerminatedStringOfRun));

  TRYODBC(hInsertStmt,
    SQL_HANDLE_STMT,
    SQLBindParameter(hInsertStmt, 2, SQL_PARAM_INPUT, SQL_C_SLONG, SQL_INTEGER, 0, 0, (SQLPOINTER)&LBACK, 0, NULL));

  // variable, n, dateTimeOfPrediction
  TRYODBC(hInsertStmt,
    SQL_HANDLE_STMT,
    SQLBindParameter(hInsertStmt, OFFSET_TO_FIRST_ACTUAL+2*OUTPUT_SIZE_I+2, SQL_PARAM_INPUT, SQL_C_CHAR, SQL_VARCHAR, 0, 0, (SQLCHAR*)VARIABLE.c_str(), 0, &nullTerminatedStringOfVariable));

  TRYODBC(hInsertStmt,
    SQL_HANDLE_STMT,
    SQLBindParameter(hInsertStmt, OFFSET_TO_FIRST_ACTUAL + 2 * OUTPUT_SIZE_I + 4, SQL_PARAM_INPUT, SQL_C_TYPE_TIMESTAMP, SQL_TYPE_TIMESTAMP, 0, 0, &now_ts, sizeof(TIMESTAMP_STRUCT), NULL));
#endif
  std::cout << "HERE2" << endl;
    
  random_device rd;     // only used once to initialise (seed) engine
  mt19937 rng(rd());    // random-number engine used (Mersenne-Twister)
  mt19937 rngForChunks(seedForChunks);
  
  vector<string> series_vect;
  unordered_map<string, M4TS> allSeries_map(30000);//max series in one chunk would be 48/2=24k, for monthly series
  unordered_map<string, string> seriesCategories_map(120000);//100k series

  std::cout << "INFO INPUT PATH: " << string(INFO_INPUT_PATH) << endl;
  ifstream infoFile(INFO_INPUT_PATH); //take in lines
  string line;
  getline(infoFile, line); //header
  while (getline(infoFile, line)) {
    //std::cout << "header" << endl;
    //cout << string( line)<<endl;

    stringstream  line_stream(line);
    string series; string category;

    getline(line_stream, series, ',');
    getline(line_stream, category, ',');
    seriesCategories_map[series] = category;
  }

  ifstream file (INPUT_PATH); //take in lines
  getline(file, line); //header
  while ( getline ( file, line) ) {
    //std::cout << "Getting line----" << endl;
    //std::cout << file << endl;
    //std::cout << line << endl;

    stringstream  line_stream(line);
    string series0;  string series;
    getline(line_stream, series0, ',' );
    for (const auto c : series0) {
      if (!ispunct(c)) {
        series.push_back(c);
      }
    }

    string category = seriesCategories_map[series]; //create a map with all time series data
    M4TS m4Obj(category, line_stream);
    if (m4Obj.n >= MIN_SERIES_LENGTH) {
      series_vect.push_back(series);
      allSeries_map[series] = m4Obj;
    }
    if (MAX_NUM_OF_SERIES>0 && series_vect.size()>=MAX_NUM_OF_SERIES)
      break;
      //std::cout << "Got line---------" << endl;
  }

  int series_len=(int)series_vect.size();
  int chunkSize= series_len/NUM_OF_CHUNKS;
  std::cout << "num of series:" << series_vect.size() <<" size of chunk:"<< chunkSize<<endl;
  uniform_int_distribution<int> uniOnSeries(0, chunkSize -1);  // closed interval [a, b]
  
  unordered_map<string, array<vector<float>, AVERAGING_LEVEL+1>> testResults_map((int)chunkSize*1.5);
  set<string> diagSeries;
  
  std::cout << "RIGHT BEFORE BIG LOOP" << endl;

  for (int ibig=0; ibig<BIG_LOOP; ibig++) { //the loop :-)
    std::cout << "BIG LOOP ITERATION: " << ibig << endl;
	int ibigDb= ibigOffset+ibig;
    string outputPath = OUTPUT_DIR + '/'+ VARIABLE + "_" + to_string(seedForChunks) + "_"+ to_string(chunkNo)+"_"+ to_string(ibigDb)+"_LB"+ to_string(LBACK)+ ".csv";
    vector<float> perfValid_vect; 
    int epochOfLastChangeOfLRate = -1;

#if defined USE_ODBC      
    TRYODBC(hInsertStmt,
      SQL_HANDLE_STMT,
      SQLBindParameter(hInsertStmt, 3, SQL_PARAM_INPUT, SQL_C_SLONG, SQL_INTEGER, 0, 0, (SQLPOINTER)&ibigDb, 0, NULL));
#endif

    ParameterCollection pc;
    ParameterCollection perSeriesPC;

    float learning_rate= INITIAL_LEARNING_RATE; //SET UP TRAINER OBJECT
    AdamTrainer trainer(pc, learning_rate, 0.9, 0.999, EPS);
    trainer.clip_threshold = GRADIENT_CLIPPING;
    AdamTrainer perSeriesTrainer(perSeriesPC, learning_rate*PER_SERIES_LR_MULTIP, 0.9, 0.999, EPS);
    perSeriesTrainer.clip_threshold = GRADIENT_CLIPPING;  
    
    #if defined USE_RESIDUAL_LSTM //MODEL CODE
      vector<ResidualDilatedLSTMBuilder> rNNStack;
      rNNStack.emplace_back(ResidualDilatedLSTMBuilder(dilations[0], INPUT_SIZE + NUM_OF_CATEGORIES, STATE_HSIZE, pc));
      for (int il = 1; il<dilations.size(); il++)
        rNNStack.emplace_back(ResidualDilatedLSTMBuilder(dilations[il], STATE_HSIZE, STATE_HSIZE, pc));
    #elif defined USE_ATTENTIVE_LSTM
      vector<AttentiveDilatedLSTMBuilder> rNNStack;
      rNNStack.emplace_back(AttentiveDilatedLSTMBuilder(dilations[0], INPUT_SIZE + NUM_OF_CATEGORIES, STATE_HSIZE, ATTENTION_HSIZE, pc));
      for (int il = 1; il<dilations.size(); il++)
        rNNStack.emplace_back(AttentiveDilatedLSTMBuilder(dilations[il], STATE_HSIZE, STATE_HSIZE, ATTENTION_HSIZE, pc));
    #else
       vector<DilatedLSTMBuilder> rNNStack;
      rNNStack.emplace_back(DilatedLSTMBuilder(dilations[0], INPUT_SIZE + NUM_OF_CATEGORIES, STATE_HSIZE, pc));
      for (int il = 1; il<dilations.size(); il++)
        rNNStack.emplace_back(DilatedLSTMBuilder(dilations[il], STATE_HSIZE, STATE_HSIZE, pc));
    #endif
    
    Parameter MLPW_par,MLPB_par;
    if (ADD_NL_LAYER) { 
      MLPW_par = pc.add_parameters({ STATE_HSIZE, STATE_HSIZE });
      MLPB_par = pc.add_parameters({ STATE_HSIZE });
    }
    Parameter adapterW_par = pc.add_parameters({ OUTPUT_SIZE, STATE_HSIZE });
    Parameter adapterB_par = pc.add_parameters({ OUTPUT_SIZE });

    shuffle(series_vect.begin(), series_vect.end(), rngForChunks);//this shuffling is psudo random (it uses the same seed) so it is synchronized across pairs of workers
    auto start= series_vect.begin()+ (chunkNo-1)*chunkSize;
    auto end= start+ chunkSize;
    if (chunkNo== NUM_OF_CHUNKS)
      end = series_vect.end();
    vector<string> oneChunk_vect(start,end);
    if (PRINT_DIAGN) {
      for (int k = 0; k<10; k++)  //diag
        cout << oneChunk_vect[k] << " ";
      cout << endl;
    }  
    if (chunkNo == NUM_OF_CHUNKS)
      cout<<"last chunk size:"<< oneChunk_vect.size()<<endl;

    std::cout << "Checking for additional parameters..." << endl;

    unordered_map<string, AdditionalParams> additionalParams_map((int)oneChunk_vect.size()*1.5); //per series
    unordered_map<string, array<AdditionalParamsF, NUM_OF_TRAIN_EPOCHS>*> historyOfAdditionalParams_map((int)oneChunk_vect.size()*1.5);
    for (auto iter = oneChunk_vect.begin(); iter != oneChunk_vect.end(); ++iter) {//setup
      string series = *iter;
      AdditionalParams addParams;
      addParams.levSm = perSeriesPC.add_parameters({ 1 }, 0.5);  //level smoothing
      addParams.sSm = perSeriesPC.add_parameters({ 1 }, 0.5);    //seasonality smoothing
      for (int isea = 0; isea<SEASONALITY; isea++)
        addParams.initSeasonality[isea] = perSeriesPC.add_parameters({ 1 }, 0.5);  //initial seasonality (over first SEASONALITY points)
      additionalParams_map[series] = addParams;

      historyOfAdditionalParams_map[series] = new array<AdditionalParamsF, NUM_OF_TRAIN_EPOCHS>();
    }
    
    std::cout << "num train epochs: " << NUM_OF_TRAIN_EPOCHS << endl;
    //BEGIN TRAINING (iterate through num_epochs)
    for (int iEpoch=0; iEpoch<NUM_OF_TRAIN_EPOCHS; iEpoch++) {
      if (!USE_AUTO_LEARNING_RATE && LEARNING_RATES.find(iEpoch) != LEARNING_RATES.end()) {
        trainer.learning_rate = LEARNING_RATES.at(iEpoch);
        perSeriesTrainer.learning_rate = LEARNING_RATES.at(iEpoch)*PER_SERIES_LR_MULTIP;
        cout << "changing LR to:" << trainer.learning_rate << endl;
      }

      vector<float> testLosses; //test losses of all series in this epoch
      vector<float> testAvgLosses; //test avg (over last few epochs) losses of all series in this epoch 
      vector<float> trainingLosses; //training losses of all series in one epoch
      vector<float> forecLosses; vector<float> levVarLosses; vector<float> stateLosses;
      #if defined USE_ODBC
      TRYODBC(hInsertStmt,
        SQL_HANDLE_STMT,
        SQLBindParameter(hInsertStmt, 5, SQL_PARAM_INPUT, SQL_C_SLONG, SQL_INTEGER, 0, 0, (SQLPOINTER)&iEpoch, 0, NULL));
      #endif

      std::cout << "entering one chunk iteration" << endl;
      
      for (auto iter = oneChunk_vect.begin() ; iter != oneChunk_vect.end(); ++iter) {
        string series=*iter;
        auto m4Obj = allSeries_map[series];

        #if defined USE_ODBC
        TRYODBC(hInsertStmt,
          SQL_HANDLE_STMT,
          SQLBindParameter(hInsertStmt, 4, SQL_PARAM_INPUT, SQL_C_CHAR, SQL_VARCHAR, 0, 0, (SQLCHAR*)series.c_str(), 0, &nullTerminatedStringOfSeries));

        TRYODBC(hInsertStmt,
          SQL_HANDLE_STMT,
          SQLBindParameter(hInsertStmt, OFFSET_TO_FIRST_ACTUAL + 2 * OUTPUT_SIZE_I + 3, SQL_PARAM_INPUT, SQL_C_SLONG, SQL_INTEGER, 0, 0, (SQLPOINTER)&m4Obj.n, 0, NULL));
        #endif
      
        ComputationGraph cg; //SET UP COMPUTATION GRAPH and load model's layers into it
         for (int il=0; il<dilations.size(); il++) {
           rNNStack[il].new_graph(cg);
           rNNStack[il].start_new_sequence(); 
         }
          
        Expression MLPW_ex, MLPB_ex;
        if (ADD_NL_LAYER) {   //Add nonlinear (?))
          MLPW_ex = parameter(cg, MLPW_par);
          MLPB_ex = parameter(cg, MLPB_par);
        }
        Expression adapterW_ex=parameter(cg, adapterW_par);
        Expression adapterB_ex=parameter(cg, adapterB_par);

        auto additionalParams= additionalParams_map[series];
        Expression levSm_ex = logistic(parameter(cg, additionalParams.levSm));  //level smoothing
		    Expression sSm_ex = logistic(parameter(cg, additionalParams.sSm)); //seasonality smoothing

			  vector<Expression> season_exVect;//vector, because we do not know how long the series is
			  for (int iseas=0; iseas<SEASONALITY; iseas++){
			    Expression seas=exp(parameter(cg, additionalParams.initSeasonality[iseas]));
			    //so, when additionalParams_map[series].initSeasonality[iseas]==0 => seas==1
			    season_exVect.push_back(seas);//Expression is a simple struct, without any storage management, so the auto copy constructor works OK.
			  }
			  season_exVect.push_back(season_exVect[0]);

			  vector<Expression> logDiffOfLevels_vect;
        vector<Expression> levels_exVect;
			  Expression lev=cdiv(input(cg, m4Obj.vals[0]), season_exVect[0]);
			  levels_exVect.push_back(lev);
        for (int i=1; i<m4Obj.vals.size();i++) {  //Exponential Smoothing-style deseasonalization and smoothing
			    Expression newLevel_ex=m4Obj.vals[i]*cdiv(levSm_ex,season_exVect[i]) + (1-levSm_ex)*levels_exVect[i-1];
			    levels_exVect.push_back(newLevel_ex);
			    Expression diff_ex=log(cdiv(newLevel_ex,levels_exVect[i-1]));//penalty for wiggliness of level
			    logDiffOfLevels_vect.push_back(diff_ex);

			    Expression newSeason_ex=m4Obj.vals[i]*cdiv(sSm_ex,newLevel_ex) + (1-sSm_ex)*season_exVect[i];
			    season_exVect.push_back(newSeason_ex);
        }
         
        Expression levelVarLoss_ex;
        if (LEVEL_VARIABILITY_PENALTY > 0) {
          vector<Expression> levelVarLoss_v; //calculate loss from each level during-
          for (int i = 1; i<logDiffOfLevels_vect.size(); i++) {
            Expression diff_ex = logDiffOfLevels_vect[i] - logDiffOfLevels_vect[i - 1];
            levelVarLoss_v.push_back(diff_ex*diff_ex);
          }
          levelVarLoss_ex = average(levelVarLoss_v);
        }

			  //if prediction horizon is larger than seasonality, so we need to repeat some of the seasonality factors
			  if (OUTPUT_SIZE_I>SEASONALITY) {
			    unsigned long startSeasonalityIndx=season_exVect.size()-SEASONALITY;
			    for (int i=0;i<(OUTPUT_SIZE_I-SEASONALITY);i++)
			      season_exVect.push_back(season_exVect[startSeasonalityIndx+i]);
			  }
        vector<Expression> losses;
        for (int i=INPUT_SIZE_I-1; i<(m4Obj.n- OUTPUT_SIZE_I); i++) { 
			    vector<Expression>::const_iterator firstE = season_exVect.begin() +i+1-INPUT_SIZE_I;
			    vector<Expression>::const_iterator pastLastE = season_exVect.begin() +i+1; //not including the last one
			    vector<Expression> inputSeasonality_exVect(firstE, pastLastE);  //[first,pastLast)
			    Expression inputSeasonality_ex=concatenate(inputSeasonality_exVect);

          vector<float>::const_iterator first = m4Obj.vals.begin() +i+1-INPUT_SIZE_I;
          vector<float>::const_iterator pastLast = m4Obj.vals.begin() +i+1; //not including the last one
          vector<float> input_vect(first, pastLast); //[first,pastLast)
          Expression input0_ex=input(cg,{INPUT_SIZE},input_vect);
			    Expression input1_ex=cdiv(input0_ex,inputSeasonality_ex); //deseasonalization
          vector<Expression> joinedInput_ex;
          input1_ex= cdiv(input1_ex, levels_exVect[i]);
          joinedInput_ex.emplace_back(noise(squash(input1_ex), NOISE_STD)); //normalization+noise
          joinedInput_ex.emplace_back(input(cg, { NUM_OF_CATEGORIES }, m4Obj.categories_vect));
          Expression input_ex = concatenate(joinedInput_ex);

          //MAKE RESNET STYLE EXPRESSION
          //std::cout << "generating resnet-style expression..." << endl;
          Expression rnn_ex;
          try {
            rnn_ex = rNNStack[0].add_input(input_ex);
            for (int il=1; il<dilations.size(); il++)
              rnn_ex=rnn_ex+rNNStack[il].add_input(rnn_ex); //resNet-style
          }  catch (exception& e) {
            cerr<<"cought exception 2 while doing "<<series<<endl;
            cerr << e.what() << endl;
            cerr <<as_vector(input_ex.value())<<endl;
          }
          Expression out_ex;
          if (ADD_NL_LAYER) {
            out_ex=MLPW_ex*rnn_ex+MLPB_ex;
            out_ex = adapterW_ex*tanh(out_ex)+adapterB_ex;
          } else 
            out_ex=adapterW_ex*rnn_ex+adapterB_ex;

          //labels
			    firstE = season_exVect.begin() +i+1;
			    pastLastE = season_exVect.begin() +i+1+OUTPUT_SIZE_I;
			    vector<Expression> outputSeasonality_exVect(firstE, pastLastE);  //[first,pastLast)
			    Expression outputSeasonality_ex=concatenate(outputSeasonality_exVect);

          first = m4Obj.vals.begin() +i+1;
          pastLast = m4Obj.vals.begin() +i+1+OUTPUT_SIZE_I;
          vector<float> labels_vect(first, pastLast);  //[first,pastLast)
          Expression labels0_ex=input(cg,{OUTPUT_SIZE},labels_vect);
			    Expression labels1_ex=cdiv(labels0_ex,outputSeasonality_ex); //deseasonalization
          labels1_ex= cdiv(labels1_ex, levels_exVect[i]);//normalization
			    Expression labels_ex=squash(labels1_ex);

          Expression loss_ex=pinBallLoss(out_ex, labels_ex);
          if (i>=INPUT_SIZE_I+MIN_INP_SEQ_LEN)
            losses.push_back(loss_ex);  
        }
        
        std::cout << "In epoch: " << iEpoch << endl;
        //std::cout << "Will begin computing frorecast loss" << endl;
        Expression forecLoss_ex= average(losses);
			  Expression loss_exp = forecLoss_ex;

        float levVarLoss=0;
        if (LEVEL_VARIABILITY_PENALTY > 0) {
          Expression levelVarLossP_ex = levelVarLoss_ex*LEVEL_VARIABILITY_PENALTY;
          levVarLoss = as_scalar(levelVarLossP_ex.value());
          levVarLosses.push_back(levVarLoss);
          loss_exp= loss_exp + levelVarLossP_ex;
        }

        float cStateLoss=0;
        if (C_STATE_PENALTY>0) {
          vector<Expression> cStateLosses_vEx;
          for (int irnn = 0; irnn < rNNStack.size(); irnn++)
            for (int it = 0; it<rNNStack[irnn].c.size(); it++) {  //first index is time
              auto& state_ex = rNNStack[irnn].c[it][0]; //c-state of first layer in a chunk at time it
              Expression penalty_ex = square(state_ex);
              cStateLosses_vEx.push_back(sum_elems(penalty_ex));
            }
          Expression cStateLossP_ex = average(cStateLosses_vEx)*C_STATE_PENALTY;
          cStateLoss = as_scalar(cStateLossP_ex.value());
          stateLosses.push_back(cStateLoss);
          loss_exp = loss_exp + cStateLossP_ex;
        }
          
        float loss = as_scalar(cg.forward(loss_exp));
        trainingLosses.push_back(loss);//losses of all series in one epoch

        float forecastLoss = loss - levVarLoss - cStateLoss;
        forecLosses.push_back(forecastLoss);
        std::cout << "forecast loss: " << forecastLoss << endl;

        //BACKPROP-TYPE THING?
        cg.backward(loss_exp);
        try {
          trainer.update();//update shared weights
          perSeriesTrainer.update();  //apdate params of this series only
          std::cout << "TRAINER UPDATED" << endl;
        } catch (exception& e) {  //long diagnostics for this unlikely event :-)
          cerr<<"cought exception while doing "<<series<<endl;
          cerr << e.what() << endl;

            float minSeason = BIG_FLOAT;
            cout << "season:";
            for (int isea = 0; isea < season_exVect.size(); isea++) {
              float val = as_scalar(season_exVect[isea].value());
              //cout << " " << val;
              if (val<minSeason)
                minSeason = val;
            }

            float minLevel = BIG_FLOAT;
            cout << "levels:";
            for (int isea = 0; isea < levels_exVect.size(); isea++) {
              float val = as_scalar(levels_exVect[isea].value());
              //cout << " " << val;
              if (val<minLevel)
                minLevel = val;
            }

            float maxAbs = 0; int timeOfMax = 0; int layerOfMax = 0; int chunkOfMax = 0;
            for (int irnn = 0; irnn < rNNStack.size(); irnn++) {
              auto state_vEx = rNNStack[irnn].c;//(time,layers)
              for (int it = 0; it < state_vEx.size(); it++) {  //through time
                for (int il = 0; il < state_vEx[it].size(); il++) {//through layers. Each layer has two states: c and h
                  auto state = as_vector(state_vEx[it][il].value());
                  for (int iv = 0; iv < state.size(); iv++) {
                    if (abs(state[iv]) > maxAbs) {
                      maxAbs = abs(state[iv]);
                      timeOfMax = it;
                      layerOfMax = il;
                      chunkOfMax = irnn;
                    }
                  }
                } //through layers/states
              } //through time
            }  //through chunks

            cout << "levSm:" << as_scalar(levSm_ex.value()) << endl;
            cout << "sSm:" << as_scalar(sSm_ex.value()) << endl;
            cout << " min season=" << minSeason << endl;
            cout << " min level=" << minLevel << endl;
            cout << " max abs:" << maxAbs << " at time:" << timeOfMax << " at layer:" << layerOfMax << " and chunk:" << chunkOfMax << endl;

            //diagSeries.insert(series);
          pc.reset_gradient();
          perSeriesPC.reset_gradient();
        }

        std::cout << "saving per-series values for diagnostic purposes" << endl;
        //saving per-series values for diagnostics purposes
        AdditionalParamsF &histAdditionalParams= historyOfAdditionalParams_map[series]->at(iEpoch);
        histAdditionalParams.levSm=as_scalar(levSm_ex.value());
        histAdditionalParams.sSm=as_scalar(sSm_ex.value());
			  for (int isea=0; isea<SEASONALITY; isea++)
			    histAdditionalParams.initSeasonality[isea]=as_scalar(season_exVect[isea].value());    
		    if (iEpoch==1 || iEpoch == NUM_OF_TRAIN_EPOCHS /2 || iEpoch == NUM_OF_TRAIN_EPOCHS-1)
          for (int iv = 0; iv<m4Obj.vals.size(); iv++) {
            histAdditionalParams.levels.push_back(as_scalar(levels_exVect[iv].value()));
            histAdditionalParams.seasons.push_back(as_scalar(season_exVect[iv].value()));
          }
          
        std::cout << "Right before testing..." << endl;
        //TEST. We walk (without learning) till end of the series. At the last point, the output is taken as the forecast
        for (int i=(m4Obj.n - OUTPUT_SIZE_I); i<m4Obj.n; i++) {
          vector<Expression>::const_iterator firstE = season_exVect.begin() + i + 1 - INPUT_SIZE_I;
          vector<Expression>::const_iterator pastLastE = season_exVect.begin() + i + 1; //not including the last one
          vector<Expression> inputSeasonality_exVect(firstE, pastLastE);  //[first,pastLast)
          Expression inputSeasonality_ex = concatenate(inputSeasonality_exVect);

          vector<float>::const_iterator first = m4Obj.vals.begin() + i + 1 - INPUT_SIZE_I;
          vector<float>::const_iterator pastLast = m4Obj.vals.begin() + i + 1; //not including the last one
          vector<float> input_vect(first, pastLast); //[first,pastLast)
          Expression input0_ex = input(cg, { INPUT_SIZE }, input_vect);
          Expression input1_ex = cdiv(input0_ex, inputSeasonality_ex); //deseasonalization
          vector<Expression> joinedInput_ex;
          input1_ex= cdiv(input1_ex, levels_exVect[i]);//normalization
          joinedInput_ex.emplace_back(squash(input1_ex));
          joinedInput_ex.emplace_back(input(cg, { NUM_OF_CATEGORIES }, m4Obj.categories_vect));
          Expression input_ex = concatenate(joinedInput_ex);

          Expression rnn_ex;
          try {
            std::cout << endl;
            std::cout << "Trying rnn_exp" << endl;
            rnn_ex = rNNStack[0].add_input(input_ex);
            for (int il=1; il<dilations.size(); il++)
              rnn_ex=rnn_ex+rNNStack[il].add_input(rnn_ex);
          }
          catch (exception& e) {
            cerr << "cought exception 2 while doing " << series << endl;
            cerr << e.what() << endl;
            cerr << as_vector(input_ex.value()) << endl;
          }
          if (i== m4Obj.n-1) {//make forecast
            firstE = season_exVect.begin() + i + 1;
            pastLastE = season_exVect.begin() + i + 1 + OUTPUT_SIZE_I;
            vector<Expression> outputSeasonality_exVect(firstE, pastLastE);  //[first,pastLast)
            Expression outputSeasonality_ex = concatenate(outputSeasonality_exVect);

            Expression out_ex;
            if (ADD_NL_LAYER) {
              out_ex=MLPW_ex*rnn_ex+MLPB_ex;
              out_ex = adapterW_ex*tanh(out_ex)+adapterB_ex;
            } else 
              out_ex=adapterW_ex*rnn_ex+adapterB_ex;
            
            out_ex = cmult(expand(out_ex), outputSeasonality_ex)*levels_exVect[i];//back to original scale
            vector<float> out_vect = as_vector(out_ex.value());
	    std::cout << "out vect size: " << out_vect.size() << endl;

            if (LBACK > 0) {
              float qLoss = errorFunc(out_vect, m4Obj.testVals);
              testLosses.push_back(qLoss);
            }

            //std::cout << "out_vect: " << to_string(out_vect) << endl;

            testResults_map[series][iEpoch%AVERAGING_LEVEL] = out_vect;
	    std::cout << "HERE5 iEpoch" << iEpoch << endl;
            if (iEpoch >= AVERAGING_LEVEL) {
	      std::cout << "HERE6" << endl;
              if (USE_MEDIAN) {
		std::cout << "HERE7" << endl;
                if (testResults_map[series][AVERAGING_LEVEL].size() == 0)
                  testResults_map[series][AVERAGING_LEVEL] = out_vect; //just to initialized, to make space. The values will be overwritten
		std::cout << "VECTOR INITIALIZED" << endl;
                for (int iii = 0; iii < OUTPUT_SIZE_I; iii++) {
                  vector<float> temp_vect2;
                  for (int ii = 0; ii<AVERAGING_LEVEL; ii++)
                    temp_vect2.push_back(testResults_map[series][ii][iii]);
                  sort(temp_vect2.begin(), temp_vect2.end());
                  testResults_map[series][AVERAGING_LEVEL][iii] = temp_vect2[MIDDLE_POS_FOR_AVG];
                }
              }
              else {
		std::cout << "HERE8" << endl;
                vector<float> firstForec = testResults_map[series][0];
                testResults_map[series][AVERAGING_LEVEL] = firstForec;
		std::cout << "HELLO" << endl;
		std::cout << firstForec.size() << endl;
                for (int ii = 1; ii<AVERAGING_LEVEL; ii++) {
                  vector<float> nextForec = testResults_map[series][ii];
                  for (int iii = 0; iii<OUTPUT_SIZE_I; iii++)
                    testResults_map[series][AVERAGING_LEVEL][iii] += nextForec[iii];
                }
                for (int iii = 0; iii<OUTPUT_SIZE_I; iii++)
                  testResults_map[series][AVERAGING_LEVEL][iii] /= AVERAGING_LEVEL;
              }

              if (LBACK > 0) {
                float qLoss = errorFunc(testResults_map[series][AVERAGING_LEVEL], m4Obj.testVals);
                testAvgLosses.push_back(qLoss);
                
                #if defined USE_ODBC       //save
                TRYODBC(hInsertStmt,
                  SQL_HANDLE_STMT,
                  SQLBindParameter(hInsertStmt, OFFSET_TO_FIRST_ACTUAL + 2 * OUTPUT_SIZE_I + 1, SQL_PARAM_INPUT, SQL_C_FLOAT, SQL_FLOAT, 0, 0, (SQLPOINTER)&forecastLoss, 0, NULL));

                for (int io = 0; io < OUTPUT_SIZE_I; io++) {
                  int ipos=OFFSET_TO_FIRST_ACTUAL + 1 + 2*io;
                  TRYODBC(hInsertStmt,
                    SQL_HANDLE_STMT,
                    SQLBindParameter(hInsertStmt, ipos, SQL_PARAM_INPUT, SQL_C_FLOAT, SQL_FLOAT, 0, 0, (SQLPOINTER)&m4Obj.testVals[io], 0, NULL));

                  TRYODBC(hInsertStmt,
                    SQL_HANDLE_STMT,
                    SQLBindParameter(hInsertStmt, ipos+1, SQL_PARAM_INPUT, SQL_C_FLOAT, SQL_FLOAT, 0, 0, (SQLPOINTER)&testResults_map[series][AVERAGING_LEVEL][io], 0, NULL));
                }
                if (MAX_NUM_OF_SERIES<0)
                  TRYODBC(hInsertStmt,
                    SQL_HANDLE_STMT,
                    SQLExecute(hInsertStmt));
                #endif    
              }
            } //time to average
            std::cout << "average found" << endl;
          }//last anchor point of the series
          std::cout << "last anchor point found" << endl;
        }//through TEST loop
        std::cout << "done with TEST loop" << endl;
      }//through series
      std::cout << "done with series" << endl;

      std::cout << "-------------FINISHED LOOP----------------";
      if (iEpoch % FREQ_OF_TEST == 0) {
        float averageTrainingLoss = accumulate(trainingLosses.begin(), trainingLosses.end(), 0.0) / trainingLosses.size();

        cout << ibig << " " << iEpoch << " loss:" << averageTrainingLoss * 100;
        if (LEVEL_VARIABILITY_PENALTY > 0 || C_STATE_PENALTY > 0) {
          float averageForecLoss = accumulate(forecLosses.begin(), forecLosses.end(), 0.0) / forecLosses.size();
          cout << " forecast loss:" << averageForecLoss*100;
        }
        if (LEVEL_VARIABILITY_PENALTY > 0) {
          float averagelevVarLoss = accumulate(levVarLosses.begin(), levVarLosses.end(), 0.0) / levVarLosses.size();
          cout << " levVar loss:" << averagelevVarLoss * 100;
        }
        if (C_STATE_PENALTY > 0) {
          float averageStateLoss = accumulate(stateLosses.begin(), stateLosses.end(), 0.0) / stateLosses.size();
          cout << " state loss:" << averageStateLoss * 100;
        }

        float averageTestLoss=0;
        if (LBACK > 0) {
          float averageTestLoss = accumulate(testLosses.begin(), testLosses.end(), 0.0) / testLosses.size();
          cout<<" Test loss:" << averageTestLoss;
          if (iEpoch >= AVERAGING_LEVEL) {
            float averageTestAvgLoss = accumulate(testAvgLosses.begin(), testAvgLosses.end(), 0.0) / testAvgLosses.size();//of this epoch
            cout << " avgLoss:" << averageTestAvgLoss;
          }
          if (USE_AUTO_LEARNING_RATE)
            perfValid_vect.push_back(averageTestLoss);
        }
        cout << endl;
      }
      
      if (USE_AUTO_LEARNING_RATE) {
        bool changeL2Rate = false;
        if (iEpoch >= 2) {
          if (iEpoch < L3_PERIOD)
            changeL2Rate = perfValid_vect[perfValid_vect.size() - 2]<LR_TOLERANCE_MULTIP*perfValid_vect[perfValid_vect.size() - 1];
          else
            changeL2Rate = perfValid_vect[perfValid_vect.size() - L3_PERIOD - 1]<LR_TOLERANCE_MULTIP*perfValid_vect[perfValid_vect.size() - 1];
        }

        if (changeL2Rate && learning_rate > MIN_LEARNING_RATE && (iEpoch - epochOfLastChangeOfLRate) >= MIN_EPOCHS_BEFORE_CHANGING_LRATE) {
          learning_rate /= LR_RATIO;
          cout << "decreasing LR to:" << learning_rate << endl;
          epochOfLastChangeOfLRate = iEpoch;
          trainer.learning_rate = learning_rate;
        }
      }
      #if defined USE_ODBC 
      TRYODBC(hDbc,
        SQL_HANDLE_DBC,
        SQLEndTran(
          SQL_HANDLE_DBC,
          hDbc,
          SQL_COMMIT));
      #endif    
    }//through epochs

    if (PRINT_DIAGN) {//some diagnostic info
      set<string> diagSeries;
      for (int i = 0; i<1; i++) {//add a few normal ones
        int irand = uniOnSeries(rng);
        diagSeries.insert(oneChunk_vect[irand]);
      }
      for (auto series : diagSeries) {
        cout << endl << series << endl;
        array<AdditionalParamsF, NUM_OF_TRAIN_EPOCHS>* historyOfAdditionalParams_ptrToArr = historyOfAdditionalParams_map[series];
        cout << "lSm:" << endl;
        for (int iEpoch = 0; iEpoch<NUM_OF_TRAIN_EPOCHS; iEpoch++)
          cout << historyOfAdditionalParams_ptrToArr->at(iEpoch).levSm << " ";
        cout << endl;
        cout << "sSm:" << endl;
        for (int iEpoch = 0; iEpoch<NUM_OF_TRAIN_EPOCHS; iEpoch++)
          cout << historyOfAdditionalParams_ptrToArr->at(iEpoch).sSm << " ";
        cout << endl;
        cout << "seasons:" << endl;
        for (int isea = 0; isea<SEASONALITY; isea++) {
          for (int iEpoch = 0; iEpoch<NUM_OF_TRAIN_EPOCHS; iEpoch++)
            cout << historyOfAdditionalParams_ptrToArr->at(iEpoch).initSeasonality[isea] << " ";
          cout << endl;
        }
        cout << endl;
        for (int iEpoch = 0; iEpoch<NUM_OF_TRAIN_EPOCHS; iEpoch++) {
          if (historyOfAdditionalParams_ptrToArr->at(iEpoch).levels.size()>0) {
            cout << "levels:" << iEpoch << " ";
            for (int iv = 0; iv<historyOfAdditionalParams_ptrToArr->at(iEpoch).levels.size(); iv++)
              cout << historyOfAdditionalParams_ptrToArr->at(iEpoch).levels[iv] << ", ";
            cout << endl;
            cout << "seas:" << iEpoch << " ";
            for (int iv = 0; iv<historyOfAdditionalParams_ptrToArr->at(iEpoch).seasons.size(); iv++)
              cout << historyOfAdditionalParams_ptrToArr->at(iEpoch).seasons[iv] << ", ";
            cout << endl;
          }
        }
      }
    }
    
    std::cout << "starting forecast save" << endl;
    //save the forecast to outputFile
    ofstream outputFile;
    outputFile.open(outputPath);
    for (auto iter = oneChunk_vect.begin(); iter != oneChunk_vect.end(); ++iter) {
      string series = *iter;
      outputFile<< series;
      for (int io=0; io<OUTPUT_SIZE_I; io++) {
	std::cout << "io: " << io << endl;
	std::cout << series << endl;
//	if (testResults_map.find(series) == testResults_map.end()){
//	  std::cout << "series not found in test results map :(" << endl;
//	}
	std::cout << testResults_map[series].size() << endl;
	std::cout << "take0" << endl;
	std::cout << AVERAGING_LEVEL << endl;
	std::cout << "take1" << endl;
	std::cout << testResults_map[series][AVERAGING_LEVEL].size() << endl;

	for (std::pair<string, array<vector<float>, AVERAGING_LEVEL+1>> element : testResults_map) {
	  std::cout << element.first << " size: " << element.second.size() << endl;
	  std::cout << element.second[0].size() << endl;
	  std::cout << element.second[1].size() << endl;
	  std::cout << element.second[2].size() << endl;
	  std::cout << element.second[3].size() << endl;
	  std::cout << element.second[4].size() << endl;
	  std::cout << element.second[5].size() << endl;
	}
        outputFile << ", "<< testResults_map[series][AVERAGING_LEVEL][io];
      }
      outputFile<<endl;
    }
    outputFile.close();


    //delete
    for (auto iter = oneChunk_vect.begin(); iter != oneChunk_vect.end(); ++iter) {
      string series = *iter;
      auto addHistArr_ptr= historyOfAdditionalParams_map[series];
      delete addHistArr_ptr;
    }
  }//ibig
  std::cout << "DONE WITH MAIN" << endl;
}//main

#if defined USE_ODBC
  #if defined _WINDOWS
	void HandleDiagnosticRecord(SQLHANDLE      hHandle,
	  SQLSMALLINT    hType,
	  RETCODE        RetCode)
	{
	  SQLSMALLINT iRec = 0;
	  SQLINTEGER  iError;
	  WCHAR       wszMessage[1000];
	  WCHAR       wszState[SQL_SQLSTATE_SIZE + 1];


	  if (RetCode == SQL_INVALID_HANDLE)
	  {
		fwprintf(stderr, L"Invalid handle!\n");
		return;
	  }

	  while (SQLGetDiagRec(hType,
		hHandle,
		++iRec,
		wszState,
		&iError,
		wszMessage,
		(SQLSMALLINT)(sizeof(wszMessage) / sizeof(WCHAR)),
		(SQLSMALLINT *)NULL) == SQL_SUCCESS)
	  {
		  fwprintf(stderr, L"[%5.5s] %s (%d)\n", wszState, wszMessage, iError);
	  }
	}
  #else
	void HandleDiagnosticRecord(SQLHANDLE      hHandle,
	  SQLSMALLINT    hType,
	  RETCODE        RetCode)
	{
	  SQLSMALLINT iRec = 0;
	  SQLINTEGER  iError;
	  SQLCHAR       wszMessage[1000];
	  SQLCHAR       wszState[SQL_SQLSTATE_SIZE + 1];


	  if (RetCode == SQL_INVALID_HANDLE)
	  {
		fwprintf(stderr, L"Invalid handle!\n");
		return;
	  }

	  while (SQLGetDiagRec(hType,
		hHandle,
		++iRec,
		wszState,
		&iError,
		wszMessage,
		1000,
		NULL) == SQL_SUCCESS)
	  {
		  fwprintf(stderr, L"[%5.5s] %s (%d)\n", wszState, wszMessage, iError);
	  }
	}
  #endif
#endif
