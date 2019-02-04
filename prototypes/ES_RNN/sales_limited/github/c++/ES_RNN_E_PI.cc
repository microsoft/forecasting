/*ES-RNN-E: Exponential Smoothing Recurrent Neural Network hybrid, Ensemble of specialists. Prediction Intervals forecast.
Slawek Smyl,  Jan-May 2017.

Dilated LSTMs, with optional shortcuts, attention. Non-seasonal, single, or double seasonal.
It is meant to be used for all types of series from M4 competition, except Monthly and Quarterly (for performance reasons - Ensamble of Specilists is slower).
The program uses and requires Dynet NN library(https://github.com/clab/dynet); can be compiled and run on Windows, Linux, and Mac.

In contradistinction to ES-RNN, each executable uses all series, but in a similar manner repeating the whole learning process BIG_LOOP times (by default 3).
Invocation should pass BIG_LOOP offset
so e.g. create a script with following lines on Windows
start <this_executable> 0
start <this_executable> 10
start <this_executable> 20
start <this_executable> 30
on 4-core computer.
In this setup, learning and fitting would be repeated 4*3 times, probably unnecessarily too many, 6-8 independent runs should be enough for a good ensemble.
Therefore if running on say 8 core machine , one can extend the above script to 8 concurrent executions and reduce BIG_LOOP to 1.
(Creating final forecasts is done in a supplied R script)

There are four blocks of parameters below, one active (starting with //PARAMS--------------) and three inactive.
These blocks are as they were during the final forecasting run. You need comment/uncomment to have one block of interest active.
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
//#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>  
#include <math.h> 

using namespace std;
using namespace dynet;

string DATA_DIR = "\\home\\deegup\\M4\\Dataset\\Train\\"; //with the competition data csvs
//string DATA_DIR="/home/uber/progs/data/M4DataSet/";
string OUTPUT_DIR = "\\home\\deegup\\M4\\output\\"; 
//string OUTPUT_DIR="/home/uber/progs/data/M4/";

int LBACK = 0; //LBACK 0 means final mode: learning on all data and forecasting. LBACK=1 would move back by OUTPUT_SIZE, and forecast last known OUTPUT_SIZE points, for backtesting. LBACK could be a larger integer, but then number of series shrinks.


//PARAMS--------------

string VARIABLE = "Hourly";
const string run0 = "(1,4)(24,168) LR=0.01, {25,3e-3f} EPOCHS=37, LVP=10, CSP=0";
const string runL = "alpha5L " + run0;
const string runH = "alpha5H " + run0;

//#define USE_RESIDUAL_LSTM
//#define USE_ATTENTIVE_LSTM
const bool ADD_NL_LAYER = false;

const int SEASONALITY_NUM = 2;//0 means no seasonality, for Yearly; 1 - single seasonality for Daily(7), Weekly(52); 2 - dual seaonality for Hourly (24,168)
const int SEASONALITY = 24;
const int SEASONALITY2 = 168;
vector<vector<unsigned>> dilations = { { 1,4 },{ 24, 168 } };

const float INITIAL_LEARNING_RATE = 0.01f;
const map<int, float> LEARNING_RATES = { { 20,1e-3f } }; //at which epoch we manually set them up to what
const float PER_SERIES_LR_MULTIP = 1;
const int NUM_OF_TRAIN_EPOCHS = 37;

float LEVEL_VARIABILITY_PENALTY = 10;  //Multiplier for L" penalty against wigglines of level vector.
const float C_STATE_PENALTY = 0;

const unsigned int STATE_HSIZE = 40;

const unsigned int INPUT_SIZE = 24;
const unsigned int OUTPUT_SIZE = 48;

const int MIN_INP_SEQ_LEN = 0;
const int MIN_SERIES_LENGTH = OUTPUT_SIZE + INPUT_SIZE + MIN_INP_SEQ_LEN + 2;  //this is compared to n==(total length - OUTPUT_SIZE). Total length may be truncated by LBACK
const int MAX_SERIES_LENGTH = 53 * SEASONALITY2 + MIN_SERIES_LENGTH;  //==all
const int TOPN = 4;


/*
string VARIABLE = "Weekly";
const string run0 = "Att 4/5 (1,52) LR=1e-3 {15,3e-4f} EPOCHS=31, LVP=100 6y";
const string runL = "alpha5L " + run0;
const string runH = "alpha5H " + run0;

//#define USE_RESIDUAL_LSTM
#define USE_ATTENTIVE_LSTM
const bool ADD_NL_LAYER = false;

const int SEASONALITY_NUM = 1; //0 means no seasonality, for Yearly; 1 - single seasonality for Daily(7), Weekly(52); 2 - dual seaonality for Hourly (24,168)
const int SEASONALITY = 52;
const int SEASONALITY2 = 0;
vector<vector<unsigned>> dilations = { { 1, 52 } };

const float INITIAL_LEARNING_RATE = 1e-3;
const map<int, float> LEARNING_RATES = { { 15,3e-4f } }; //at which epoch we manually set them up to what
const float PER_SERIES_LR_MULTIP = 1;
const int NUM_OF_TRAIN_EPOCHS = 31;

float LEVEL_VARIABILITY_PENALTY = 100;  //Multiplier for L" penalty against wigglines of level vector. 
const float C_STATE_PENALTY = 0;

const unsigned int STATE_HSIZE = 40;

const unsigned int INPUT_SIZE = 10;
const unsigned int OUTPUT_SIZE = 13;

const int MIN_INP_SEQ_LEN = 0;
const int MIN_SERIES_LENGTH = OUTPUT_SIZE + INPUT_SIZE + MIN_INP_SEQ_LEN + 2;  //this is compared to n==(total length - OUTPUT_SIZE). Total length may be truncated by LBACK
                                                                               //#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
                                                                               //#81     380     935    1023    1604    2598
const int MAX_SERIES_LENGTH = 6 * SEASONALITY + MIN_SERIES_LENGTH;
const int TOPN = 4;
*/

/*

string VARIABLE = "Daily";
const string run0 = "4/5 (1,3)(7,14) LR=3e-4 {13,1e-4f} EPOCHS=21, LVP=100 13w";
const string runL = "alpha5L " + run0;
const string runH = "alpha5H " + run0;

//#define USE_RESIDUAL_LSTM
//#define USE_ATTENTIVE_LSTM
const bool ADD_NL_LAYER=false;

const int SEASONALITY_NUM = 1; //0 means no seasonality, for Yearly; 1 - single seasonality for Daily(7), Weekly(52); 2 - dual seaonality for Hourly (24,168)
const int SEASONALITY = 7;
const int SEASONALITY2 = 0;
vector<vector<unsigned>> dilations = { { 1,3 },{ 7, 14 } };

const float INITIAL_LEARNING_RATE = 3e-4;
const map<int, float> LEARNING_RATES = { { 13,1e-4f } }; //at which epoch we manually set them up to what
const float PER_SERIES_LR_MULTIP = 1;
const int NUM_OF_TRAIN_EPOCHS = 21;

float LEVEL_VARIABILITY_PENALTY = 100;  //Multiplier for L" penalty against wigglines of level vector. 
const float C_STATE_PENALTY = 0;

const unsigned int STATE_HSIZE = 40;

const unsigned int INPUT_SIZE = 7;
const unsigned int OUTPUT_SIZE = 14;

const int MIN_INP_SEQ_LEN = 0;
const int MIN_SERIES_LENGTH = OUTPUT_SIZE + INPUT_SIZE + MIN_INP_SEQ_LEN + 2;  //this is compared to n==(total length - OUTPUT_SIZE). Total length may be truncated by LBACK
                                                                               //#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
                                                                               //##93     323    2940    2357    4197    9919 
const int MAX_SERIES_LENGTH = 13 * SEASONALITY + MIN_SERIES_LENGTH;
const int TOPN = 4;
*/

/*
string VARIABLE = "Yearly";
const string run0 = "Att NL 4/5 (1,6) LR=1e-4 {17,3e-5}{22,1e-5} EPOCHS=29, 60*";
const string runL = "alpha5L " + run0;
const string runH = "alpha5H " + run0;

//#define USE_RESIDUAL_LSTM
#define USE_ATTENTIVE_LSTM
const bool ADD_NL_LAYER = true;

const int SEASONALITY_NUM = 0; //0 means no seasonality
const int SEASONALITY = 1; //for no seasonality, set it to 1, important
const int SEASONALITY2 = 0;
vector<vector<unsigned>> dilations = { { 1,6 } };

const float INITIAL_LEARNING_RATE = 1e-4;
const map<int, float> LEARNING_RATES = { { 17,3e-5 },{ 22,1e-5 } }; //at which epoch we manually set them up to what
const float PER_SERIES_LR_MULTIP = 1;
const int NUM_OF_TRAIN_EPOCHS = 29;

float LEVEL_VARIABILITY_PENALTY = 0;  //Multiplier for L" penalty against wigglines of level vector. 
const float C_STATE_PENALTY = 0;

const unsigned int STATE_HSIZE = 30;

const unsigned int INPUT_SIZE = 4;
const unsigned int OUTPUT_SIZE = 6;

const int MIN_INP_SEQ_LEN = 0;
const int MIN_SERIES_LENGTH = OUTPUT_SIZE + INPUT_SIZE + MIN_INP_SEQ_LEN + 2;  //this is compared to n==(total length - OUTPUT_SIZE). Total length may be truncated by LBACK
                                                                               //#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
                                                                               //#13.00   20.00   29.00   31.32   40.00  835.00
const int MAX_SERIES_LENGTH = 60 + MIN_SERIES_LENGTH;
const int TOPN = 4;
*/

const float ALPHA = 0.05;
const float TAUL = ALPHA / 2;
const float TAUH = 1 - TAUL;
const float ALPHA_MULTIP = 2 / ALPHA;

const int BIG_LOOP = 3;
const int NUM_OF_NETS = 5;
const unsigned ATTENTION_HSIZE = STATE_HSIZE;

#if defined _DEBUG
  const int MAX_NUM_OF_SERIES = 20;
#else
  const int MAX_NUM_OF_SERIES = -1;
#endif // _DEBUG

const unsigned int NUM_OF_CATEGORIES = 6;
const int AVERAGING_LEVEL = 5;
const float EPS=1e-6;

const float NOISE_STD=0.001; 
const int FREQ_OF_TEST=1;
const float GRADIENT_CLIPPING=50;
const float BIG_FLOAT=1e38;//numeric_limits<float>::max();
const bool PRINT_DIAGN = false;

string INPUT_PATH = DATA_DIR + VARIABLE + "-train.csv";
string INFO_INPUT_PATH = DATA_DIR + "M4-info.csv";


Expression squash(Expression& x) {
  return log(x);
}
float squash(float x) {
  return log(x);
}

Expression expand(Expression& x) {
  return exp(x);
}
float expand(float x) {
  return exp(x);
}


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
  float meanAbsSeasDiff;
  int n;
  
  M4TS(string category, stringstream  &line_stream) {
    array<float, NUM_OF_CATEGORIES> categories = { 0,0,0,0,0,0 };
    if (category == "Demographic")
      categories[0] = 1;
    else if (category == "Finance")
      categories[1] = 1;
    else if (category == "Industry")
      categories[2] = 1;
    else if (category == "Macro")
      categories[3] = 1;
    else if (category == "Micro")
      categories[4] = 1;
    else if (category == "Other")
      categories[5] = 1;
    else {
      cerr << "unknown category?";
      exit(-1);
    }
    for (int i = 0; i < NUM_OF_CATEGORIES; i++)
      categories_vect.push_back(categories[i]);

    string tmp_str;
    while(getline(line_stream, tmp_str, ',' )) {
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

    meanAbsSeasDiff = 0;
    float sumf = 0;
    for (int ip = SEASONALITY; ip<vals.size(); ip++) {
      float diff = vals[ip] - vals[ip - SEASONALITY];
      sumf += abs(diff);
    }
    if (sumf>0)
      meanAbsSeasDiff = sumf / (vals.size() - SEASONALITY);

    if (LBACK > 0) {  //extract last OUTPUT_SIZE points as the test values
      if (vals.size() > LBACK*OUTPUT_SIZE) {
        auto first = vals.begin() + vals.size() - LBACK*OUTPUT_SIZE;
        auto pastLast = vals.begin() + vals.size() - (LBACK-1)*OUTPUT_SIZE;
        vector<float> input_vect(first, pastLast); //[first,pastLast)
        testVals= input_vect;
        vals.resize(vals.size() - LBACK*OUTPUT_SIZE); //remove last LBACK*OUTPUT_SIZE elements
        n = vals.size();
      } else
        n = 0;
    } else {
      n = vals.size();
    }
    if (n > MAX_SERIES_LENGTH) {//chop long series
      vals.erase(vals.begin(), vals.begin() + (n-MAX_SERIES_LENGTH)); //remove some early data
      n = vals.size();
    }
  }
  M4TS(){};
};

#if defined USE_ODBC        
void HandleDiagnosticRecord(SQLHANDLE      hHandle,
  SQLSMALLINT    hType,
  RETCODE        RetCode);
#endif 

struct AdditionalParams {//Per series, important
    Parameter levSm;
    Parameter sSm;
    array<Parameter, SEASONALITY> initSeasonality;
    Parameter sSm2;
    array<Parameter, SEASONALITY2> initSeasonality2;
};
struct AdditionalParamsF {//Used for storing diagnostics
    float levSm;
    float sSm;
    array<float, SEASONALITY> initSeasonality;
    float sSm2;
    array<float, SEASONALITY2> initSeasonality2;
    vector<float> levels;
    vector<float> seasons;
    vector<float> seasons2;
};
  

array<int, NUM_OF_NETS> perfToRanking (array<float, NUM_OF_NETS> perf_arr) {
  array<int, NUM_OF_NETS> index;
  
  for (int itop=0; itop<TOPN; itop++) {
    float currMin=BIG_FLOAT; int indexOfMin=-1;
    for (int i=0; i<NUM_OF_NETS; i++) {
      if (perf_arr[i]<currMin) {
        currMin=perf_arr[i];
        indexOfMin=i;
      }
    }
    index[itop]=indexOfMin;
    perf_arr[indexOfMin]=BIG_FLOAT;
  }
  return index;
}

//loss function
Expression MSIS(const Expression& out_ex, const Expression& actuals_ex) {
  vector<Expression> losses;
  for (unsigned int indx = 0; indx<OUTPUT_SIZE; indx++) {
    auto forecL = pick(out_ex, indx);
    auto forecH = pick(out_ex, indx+ OUTPUT_SIZE);
    auto actual = pick(actuals_ex, indx);
    float actualf= as_scalar(actual.value());

    Expression loss= forecH - forecL;
    if (actualf< as_scalar(forecL.value()))
      loss=loss+(forecL - actual)*ALPHA_MULTIP;
    if (actualf > as_scalar(forecH.value()))
      loss = loss + (actual - forecH)*ALPHA_MULTIP;
    losses.push_back(loss);
  }
  return sum(losses) / OUTPUT_SIZE;
}

// weighted quantile Loss
float wQuantLoss(vector<float>& out_vect, vector<float>& actuals_vect, float tau, int offset) {//used just for diagnostics, if if LBACK>0 and PERCENTILE!=50
  float sumf = 0; float suma = 0;
  for (unsigned int indx = 0; indx<OUTPUT_SIZE; indx++) {
    auto forec = out_vect[indx+ offset];
    auto actual = actuals_vect[indx];
    suma += abs(actual);
    if (actual > forec)
      sumf = sumf + (actual - forec)*tau;
    else
      sumf = sumf + (actual - forec)*(tau - 1);
  }
  return sumf / suma * 200;
}

float errorFunc(vector<float>& out_vect, vector<float>& actuals_vect, float meanAbsSeasDiff) {
  float sumf=0;
  for (unsigned int indx = 0; indx<OUTPUT_SIZE; indx++) {
    auto forecL = out_vect[indx];
    auto forecH = out_vect[indx + OUTPUT_SIZE];
    auto actualf = actuals_vect[indx];

    float loss = forecH - forecL;
    if (actualf< forecL)
      loss = loss + (forecL - actualf)*ALPHA_MULTIP;
    if (actualf > forecH)
      loss = loss + (actualf - forecH)*ALPHA_MULTIP;
    sumf+=loss;
  }
  return sumf / (OUTPUT_SIZE*meanAbsSeasDiff);
}



int main(int argc, char** argv) {
  dynet::initialize(argc, argv);

  int ibigOffset = 0;
  if (argc == 2)
    ibigOffset = atoi(argv[1]);
    
  cout<<VARIABLE<<" "<<runL<<endl;
  cout << runH << " Lback=" << LBACK << endl;
  cout << "ibigOffset:"<< ibigOffset<<endl;

  if (SEASONALITY_NUM <= 0 && LEVEL_VARIABILITY_PENALTY > 0) {
    cout<<"Warning. LEVEL_VARIABILITY_PENALTY has to be equal zero if SEASONALITY_NUM==0"<<endl;
    LEVEL_VARIABILITY_PENALTY=0;
  }
  
  time_t rawtime;
  struct tm * timeinfo;
  char buffer[80];

  time(&rawtime);
  timeinfo = localtime(&rawtime);

  strftime(buffer, sizeof(buffer), "%Y-%m-%d_%I_%M", timeinfo);
  std::string timestamp_str(buffer);

  ostringstream convert2;
  convert2 << int(ALPHA * 100);

  #if defined _WINDOWS
    OUTPUT_DIR = OUTPUT_DIR + "\\" + VARIABLE+ timestamp_str;
    if (LBACK==0) 
      OUTPUT_DIR = OUTPUT_DIR+"Final\\";
    OUTPUT_DIR = OUTPUT_DIR + convert2.str();
    string exec = string("mkdir ") + OUTPUT_DIR;//so occasionaly, if the programs do not start within the same minute, you may find more than one output dir created. After the run just manullay put them together.
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
  for (int iq = 1; iq <= OUTPUT_SIZE; iq++) {
    stringstream ss;
    ss << iq;
    string iq_str = ss.str();
    insertQuery_str = insertQuery_str +", actual"+iq_str+", forec" + iq_str;
  }
  insertQuery_str = insertQuery_str +", trainingError, variable, n, dateTimeOfPrediction) \
    values(? , ? , ? , ? , ? ";
  for (int iq = 1; iq <= OUTPUT_SIZE; iq++) {
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
    SQLBindParameter(hInsertStmt, 2, SQL_PARAM_INPUT, SQL_C_SLONG, SQL_INTEGER, 0, 0, (SQLPOINTER)&LBACK, 0, NULL));

  // variable, n, dateTimeOfPrediction
  TRYODBC(hInsertStmt,
    SQL_HANDLE_STMT,
    SQLBindParameter(hInsertStmt, OFFSET_TO_FIRST_ACTUAL+2*OUTPUT_SIZE+2, SQL_PARAM_INPUT, SQL_C_CHAR, SQL_VARCHAR, 0, 0, (SQLCHAR*)VARIABLE.c_str(), 0, &nullTerminatedStringOfVariable));

  TRYODBC(hInsertStmt,
    SQL_HANDLE_STMT,
    SQLBindParameter(hInsertStmt, OFFSET_TO_FIRST_ACTUAL + 2 * OUTPUT_SIZE + 4, SQL_PARAM_INPUT, SQL_C_TYPE_TIMESTAMP, SQL_TYPE_TIMESTAMP, 0, 0, &now_ts, sizeof(TIMESTAMP_STRUCT), NULL));
#endif
    
  random_device rd;     // only used once to initialise (seed) engine
  mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
  
  vector<string> series_vect;
  unordered_map<string, M4TS> allSeries_map(30000);//max series in one chunk would be 24k for yearly series
  unordered_map<string, string> seriesCategories_map(120000);//100k series

  ifstream infoFile(INFO_INPUT_PATH);
  string line;
  getline(infoFile, line); //header
  while (getline(infoFile, line)) {
    //cout << string( line)<<endl;
    stringstream  line_stream(line);
    string series; string category;

    getline(line_stream, series, ',');
    getline(line_stream, category, ',');
    seriesCategories_map[series] = category;
  }

  ifstream file (INPUT_PATH);
  getline(file, line); //header
  while ( getline ( file, line) ) {
    stringstream  line_stream(line);
    string series0;  string series;
    getline(line_stream, series0, ',' );
    for (const auto c : series0) {
      if (!ispunct(c)) {
        series.push_back(c);
      }
    }

    string category = seriesCategories_map[series];
    M4TS m4Obj(category, line_stream);
    if (m4Obj.n >= MIN_SERIES_LENGTH) {
      series_vect.push_back(series);
      if (m4Obj.meanAbsSeasDiff==0) {
        cout<<"Warning, flat series:"<<series<<endl;
        m4Obj.meanAbsSeasDiff= m4Obj.testVals[0]/100;
      }
      allSeries_map[series] = m4Obj;
    }
    if (MAX_NUM_OF_SERIES>0 && series_vect.size()>=MAX_NUM_OF_SERIES)
      break;
  }
  cout << "num of series:" << series_vect.size() << endl;

  unsigned int series_len=(unsigned int)series_vect.size();
  uniform_int_distribution<int> uniOnSeries(0,series_len-1);  // closed interval [a, b]
  uniform_int_distribution<int> uniOnNets(0,NUM_OF_NETS-1);  // closed interval [a, b]
  
  unordered_map<string, array<array<vector<float>, AVERAGING_LEVEL+1>, NUM_OF_NETS>> testResults_map((int)series_len*1.5);//per series, etc...
  unordered_map<string, vector<float>> finalResults_map((int)series_len*1.5);//per series
  set<string> diagSeries;
  
  unordered_map<string, array<int, NUM_OF_NETS>> netRanking_map;
  for (int ibig=0; ibig<BIG_LOOP; ibig++) {
  	int ibigDb= ibigOffset+ibig;
    string outputPathL = OUTPUT_DIR + '/'+ VARIABLE + "_" + to_string(ibigDb)+"_LLB"+ to_string(LBACK)+ ".csv";
    string outputPathH = OUTPUT_DIR + '/' + VARIABLE + "_" + to_string(ibigDb) + "_HLB" + to_string(LBACK) + ".csv";
    vector<float> perfValid_vect; 
    int epochOfLastChangeOfLRate = -1;
    
#if defined USE_ODBC        
    TRYODBC(hInsertStmt,
      SQL_HANDLE_STMT,
      SQLBindParameter(hInsertStmt, 3, SQL_PARAM_INPUT, SQL_C_SLONG, SQL_INTEGER, 0, 0, (SQLPOINTER)&ibigDb, 0, NULL));
#endif 
  
    //create nets
    array<ParameterCollection, NUM_OF_NETS> paramsCollection_arr;//per net
    array<ParameterCollection, NUM_OF_NETS> perSeriesParamsCollection_arr;//per net
    array<AdamTrainer*, NUM_OF_NETS> trainers_arr;
    array<AdamTrainer*, NUM_OF_NETS> perSeriesTrainers_arr;
    

    #if defined USE_RESIDUAL_LSTM
      array<vector<ResidualDilatedLSTMBuilder>, NUM_OF_NETS> rnnStack_arr;
    #elif defined USE_ATTENTIVE_LSTM
      array<vector<AttentiveDilatedLSTMBuilder>, NUM_OF_NETS> rnnStack_arr;
    #else
      array<vector<DilatedLSTMBuilder>, NUM_OF_NETS> rnnStack_arr;
    #endif

    array<Parameter, NUM_OF_NETS> MLPW_parArr;
    array<Parameter, NUM_OF_NETS> MLPB_parArr;
    array<Parameter, NUM_OF_NETS> adapterW_parArr;
    array<Parameter, NUM_OF_NETS> adapterB_parArr;
    
    //this is not a history, this is the real stuff
    unordered_map<string, array<AdditionalParams, NUM_OF_NETS>* > additionalParams_mapOfArr((int)series_len*1.5); //per series, per net
    for (auto iter = series_vect.begin() ; iter != series_vect.end(); ++iter) {
      string series=*iter;
      additionalParams_mapOfArr[series]=new array<AdditionalParams, NUM_OF_NETS>();
    }
    
    for (int inet=0; inet<NUM_OF_NETS; inet++) {
      ParameterCollection& pc=paramsCollection_arr[inet];
      ParameterCollection& perSeriesPC=perSeriesParamsCollection_arr[inet];
      
      trainers_arr[inet]=new AdamTrainer (pc, INITIAL_LEARNING_RATE, 0.9, 0.999, EPS);
      trainers_arr[inet]->clip_threshold = GRADIENT_CLIPPING;
      perSeriesTrainers_arr[inet]=new AdamTrainer (perSeriesPC, INITIAL_LEARNING_RATE*PER_SERIES_LR_MULTIP, 0.9, 0.999, EPS);
      perSeriesTrainers_arr[inet]->clip_threshold = GRADIENT_CLIPPING;
            
    auto& rNNStack=rnnStack_arr[inet];
    #if defined USE_RESIDUAL_LSTM
      rNNStack.emplace_back(ResidualDilatedLSTMBuilder(dilations[0], INPUT_SIZE + NUM_OF_CATEGORIES, STATE_HSIZE, pc));
      for (int il = 1; il<dilations.size(); il++)
        rNNStack.emplace_back(ResidualDilatedLSTMBuilder(dilations[il], STATE_HSIZE, STATE_HSIZE, pc));
    #elif defined USE_ATTENTIVE_LSTM
      rNNStack.emplace_back(AttentiveDilatedLSTMBuilder(dilations[0], INPUT_SIZE + NUM_OF_CATEGORIES, STATE_HSIZE, ATTENTION_HSIZE, pc));
      for (int il = 1; il<dilations.size(); il++)
        rNNStack.emplace_back(AttentiveDilatedLSTMBuilder(dilations[il], STATE_HSIZE, STATE_HSIZE, ATTENTION_HSIZE, pc));
    #else
      rNNStack.emplace_back(DilatedLSTMBuilder(dilations[0], INPUT_SIZE + NUM_OF_CATEGORIES, STATE_HSIZE, pc));
      for (int il = 1; il<dilations.size(); il++)
        rNNStack.emplace_back(DilatedLSTMBuilder(dilations[il], STATE_HSIZE, STATE_HSIZE, pc));
    #endif
    
      if (ADD_NL_LAYER) { 
        MLPW_parArr[inet] = pc.add_parameters({ STATE_HSIZE, STATE_HSIZE });
        MLPB_parArr[inet] = pc.add_parameters({ STATE_HSIZE });
      }
  	  adapterW_parArr[inet]=pc.add_parameters({OUTPUT_SIZE*2, STATE_HSIZE});
  	  adapterB_parArr[inet]=pc.add_parameters({OUTPUT_SIZE*2});
      
      for (auto iter = series_vect.begin() ; iter != series_vect.end(); ++iter) {
        string series=*iter;
        array<AdditionalParams, NUM_OF_NETS>*  additionalParams_arr=additionalParams_mapOfArr[series];
        additionalParams_arr->at(inet).levSm=perSeriesPC.add_parameters({1}, 0.5);//per series, per net
        if (SEASONALITY_NUM > 0) {
          additionalParams_arr->at(inet).sSm = perSeriesPC.add_parameters({ 1 }, 0.5);
          for (int isea = 0; isea<SEASONALITY; isea++)
            additionalParams_arr->at(inet).initSeasonality[isea] = perSeriesPC.add_parameters({ 1 }, 0.5);
        }
        if (SEASONALITY_NUM > 1) {
          additionalParams_arr->at(inet).sSm2 = perSeriesPC.add_parameters({ 1 }, 0.5);
          for (int isea = 0; isea<SEASONALITY2; isea++)
            additionalParams_arr->at(inet).initSeasonality2[isea] = perSeriesPC.add_parameters({ 1 }, 0.5);
        }
      }
    }//seting up, through nets
    
    //history of params. Series->[NUM_OF_NETS,NUM_OF_TRAIN_EPOCHS]
    unordered_map<string, array<array<AdditionalParamsF, NUM_OF_TRAIN_EPOCHS>, NUM_OF_NETS>*> historyOfAdditionalParams_map((int)series_len*1.5);
    for (auto iter = series_vect.begin() ; iter != series_vect.end(); ++iter) {
      string series=*iter;
      historyOfAdditionalParams_map[series]=new array<array<AdditionalParamsF, NUM_OF_TRAIN_EPOCHS>, NUM_OF_NETS>();
    }
    
    //first assignment. Yes, we are using vector , so the very first time the duplicates are possible. But a set can't be sorted
    array<vector<string>, NUM_OF_NETS> seriesAssignment;//every net has an array
    for (int j=0; j<NUM_OF_NETS/2; j++)
      for (int i=0; i<series_len; i++) {
        int inet=uniOnNets(rng);
        seriesAssignment[inet].push_back(series_vect[i]);
      }
    
    //nesting: ibig
    for (int iEpoch=0; iEpoch<NUM_OF_TRAIN_EPOCHS; iEpoch++) {
      #if defined USE_ODBC
        TRYODBC(hInsertStmt,
        SQL_HANDLE_STMT,
        SQLBindParameter(hInsertStmt, 5, SQL_PARAM_INPUT, SQL_C_SLONG, SQL_INTEGER, 0, 0, (SQLPOINTER)&iEpoch, 0, NULL));
      #endif
    
      unordered_map<string, array<float, NUM_OF_NETS>> netPerf_map;
      for (int inet=0; inet<NUM_OF_NETS; inet++) {  //Parellalize here, if you can :-)
        //initialize perf matrix
        for (auto iter = series_vect.begin() ; iter != series_vect.end(); ++iter) {
          string series=*iter;
          netPerf_map[series][inet]=BIG_FLOAT;
        }
        
        ParameterCollection& pc=paramsCollection_arr[inet];       
        auto& trainer=trainers_arr[inet];
        ParameterCollection& perSeriesPC=perSeriesParamsCollection_arr[inet];
        auto& perSeriesTrainer=perSeriesTrainers_arr[inet];
        
      	if (LEARNING_RATES.find(iEpoch) != LEARNING_RATES.end()) {
        		trainer->learning_rate = LEARNING_RATES.at(iEpoch);
        		if (inet==0)
        		  cout << "changing LR to:" << trainer->learning_rate << endl;
        		perSeriesTrainer->learning_rate = LEARNING_RATES.at(iEpoch)*PER_SERIES_LR_MULTIP;
      	}

        auto& rNNStack=rnnStack_arr[inet];
        Parameter& MLPW_par = MLPW_parArr[inet];
        Parameter& MLPB_par = MLPB_parArr[inet];
        Parameter& adapterW_par=adapterW_parArr[inet];
        Parameter& adapterB_par=adapterB_parArr[inet];
        
        vector<string> oneNetAssignments=seriesAssignment[inet];
        random_shuffle (oneNetAssignments.begin(), oneNetAssignments.end());
        
        vector<float> epochLosses;
        vector<float> forecLosses; vector<float> levVarLosses; vector<float> stateLosses;
        for (auto iter = oneNetAssignments.begin() ; iter != oneNetAssignments.end(); ++iter) {
          string series=*iter;
          auto m4Obj=allSeries_map[series];
        
          ComputationGraph cg;
          for (int il=0; il<dilations.size(); il++) {
            rNNStack[il].new_graph(cg);
            rNNStack[il].start_new_sequence(); 
          }
          
          AdditionalParams& additionalParams=additionalParams_mapOfArr[series]->at(inet);
          array<AdditionalParamsF, NUM_OF_TRAIN_EPOCHS>& historyOfAdditionalParams_arr=historyOfAdditionalParams_map[series]->at(inet);

					Expression MLPW_ex,MLPB_ex;
          if (ADD_NL_LAYER)  {
            MLPW_ex = parameter(cg, MLPW_par);
            MLPB_ex = parameter(cg, MLPB_par);
          }
          Expression adapterW_ex=parameter(cg, adapterW_par);
          Expression adapterB_ex=parameter(cg, adapterB_par);

          Expression levSmSerNet0_ex= parameter(cg, additionalParams.levSm);
          Expression levSm_ex = logistic(levSmSerNet0_ex);

          vector<Expression> season_exVect;//vector, because we do not know how long the series is
          Expression sSm_ex;
          if (SEASONALITY_NUM > 0) {
            Expression sSmSerNet0_ex= parameter(cg, additionalParams.sSm);
            sSm_ex = logistic(sSmSerNet0_ex);
            
            for (int isea = 0; isea<SEASONALITY; isea++) {
              Expression sSerNet0 = parameter(cg, additionalParams.initSeasonality[isea]);  //per series, per net
              Expression s1_ex = exp(sSerNet0);
              season_exVect.push_back(s1_ex);//Expression is a simple struct, without any storage management, so the auto copy constructor works OK.            
            }
            season_exVect.push_back(season_exVect[0]);
          }

          vector<Expression> season2_exVect;//vector, because we do not know how long the series is
          Expression sSm2_ex;
          if (SEASONALITY_NUM > 1) {
            Expression sSm2SerNet0_ex= parameter(cg, additionalParams.sSm2);
            sSm2_ex = logistic(sSm2SerNet0_ex);
            
            for (int isea = 0; isea<SEASONALITY2; isea++) {
              Expression sSer2Net0 = parameter(cg, additionalParams.initSeasonality2[isea]);  //per series, per net
              Expression s2_ex = exp(sSer2Net0);
              season2_exVect.push_back(s2_ex);//Expression is a simple struct, without any storage management, so the auto copy constructor works OK.            
            }
            season2_exVect.push_back(season2_exVect[0]);
          }

		      vector<Expression> logDiffOfLevels_vect;
          vector<Expression> levels_exVect;
          if (SEASONALITY_NUM == 0) {
            levels_exVect.push_back(input(cg, m4Obj.vals[0]));
            for (int i = 1; i<m4Obj.vals.size(); i++) {
              Expression newLevel_ex = levSm_ex*m4Obj.vals[i] + (1 - levSm_ex)*levels_exVect[i - 1];
              levels_exVect.push_back(newLevel_ex);
            }
          }
          else if (SEASONALITY_NUM == 1) {
            Expression lev = cdiv(input(cg, m4Obj.vals[0]), season_exVect[0]);
            levels_exVect.push_back(lev);
            for (int i = 1; i<m4Obj.vals.size(); i++) {//Exponential Smoothing-style deseasonalization and smoothing
              Expression newLevel_ex = m4Obj.vals[i] * cdiv(levSm_ex, season_exVect[i]) + (1 - levSm_ex)*levels_exVect[i - 1];
              levels_exVect.push_back(newLevel_ex);
              Expression diff_ex = log(cdiv(newLevel_ex, levels_exVect[i - 1])); //penalty for wiggliness of level
              logDiffOfLevels_vect.push_back(diff_ex);

              Expression newSeason_ex = m4Obj.vals[i] * cdiv(sSm_ex, newLevel_ex) + (1 - sSm_ex)*season_exVect[i];
              season_exVect.push_back(newSeason_ex);
            }

            //if prediction horizon is larger than seasonality, so we need to repeat some of the seasonality factors
            if (OUTPUT_SIZE>SEASONALITY) {
              unsigned long startSeasonalityIndx = season_exVect.size() - SEASONALITY;
              for (int i = 0; i<(OUTPUT_SIZE - SEASONALITY); i++)
                season_exVect.push_back(season_exVect[startSeasonalityIndx + i]);
            }
          }
          else if (SEASONALITY_NUM == 2) {
            Expression lev = cdiv(input(cg, m4Obj.vals[0]), season_exVect[0] * season2_exVect[0]);
            levels_exVect.push_back(lev);
            for (int i = 1; i<m4Obj.vals.size(); i++) {
              Expression newLevel_ex = m4Obj.vals[i] * cdiv(levSm_ex, season_exVect[i] * season2_exVect[i]) + (1 - levSm_ex)*levels_exVect[i - 1];
              levels_exVect.push_back(newLevel_ex);
              Expression diff_ex = log(cdiv(newLevel_ex, levels_exVect[i - 1]));
              logDiffOfLevels_vect.push_back(diff_ex);

              Expression newSeason_ex = m4Obj.vals[i] * cdiv(sSm_ex, newLevel_ex*season2_exVect[i]) + (1 - sSm_ex)*season_exVect[i];
              season_exVect.push_back(newSeason_ex);
              Expression newSeason2_ex = m4Obj.vals[i] * cdiv(sSm2_ex, newLevel_ex*season_exVect[i]) + (1 - sSm2_ex)*season2_exVect[i];
              season2_exVect.push_back(newSeason2_ex);
            }

            //if prediction horizon is larger than seasonality, so we need to repeat some of the seasonality factors
            if (OUTPUT_SIZE>SEASONALITY) {
              unsigned long startSeasonalityIndx = season_exVect.size() - SEASONALITY;
              for (int i = 0; i<(OUTPUT_SIZE - SEASONALITY); i++)
                season_exVect.push_back(season_exVect[startSeasonalityIndx + i]);
            }
            //if prediction horizon is larger than seasonality, so we need to repeat some of the seasonality factors
            if (OUTPUT_SIZE>SEASONALITY2) {
              unsigned long startSeasonalityIndx = season2_exVect.size() - SEASONALITY2;
              for (int i = 0; i<(OUTPUT_SIZE - SEASONALITY2); i++)
                season2_exVect.push_back(season2_exVect[startSeasonalityIndx + i]);
            }
          }
          else {
            cerr<<"SEASONALITY_NUM="<< SEASONALITY_NUM;
            exit(-1);
          }
		     
          Expression levelVarLoss_ex;
          if (LEVEL_VARIABILITY_PENALTY > 0) {
            vector<Expression> levelVarLoss_v;
            for (int i = 1; i<logDiffOfLevels_vect.size(); i++) {
              Expression diff_ex = logDiffOfLevels_vect[i] - logDiffOfLevels_vect[i - 1];
              levelVarLoss_v.push_back(diff_ex*diff_ex);
            }
            levelVarLoss_ex = average(levelVarLoss_v);
          }
			   
          Expression inputSeasonality_ex; Expression inputSeasonality2_ex;
          Expression outputSeasonality_ex; Expression outputSeasonality2_ex;
          vector<Expression> losses;//losses of steps through single time series
          for (int i=INPUT_SIZE-1; i<(m4Obj.n- OUTPUT_SIZE); i++) { 
            vector<float>::const_iterator first = m4Obj.vals.begin() + i + 1 - INPUT_SIZE;
            vector<float>::const_iterator pastLast = m4Obj.vals.begin() + i + 1; //not including the last one
            vector<float> input_vect(first, pastLast); //[first,pastLast)

            first = m4Obj.vals.begin() + i + 1;
            pastLast = m4Obj.vals.begin() + i + 1 + OUTPUT_SIZE;
            vector<float> labels_vect(first, pastLast);  //[first,pastLast)

            Expression input1_ex = input(cg, { INPUT_SIZE }, input_vect);
            Expression labels1_ex = input(cg, { OUTPUT_SIZE }, labels_vect);

            if (SEASONALITY_NUM > 0 ) {
			        vector<Expression>::const_iterator firstE = season_exVect.begin() +i+1-INPUT_SIZE;
			        vector<Expression>::const_iterator pastLastE = season_exVect.begin() +i+1; //not including the last one
			        vector<Expression> inputSeasonality_exVect(firstE, pastLastE);  //[first,pastLast)
			        inputSeasonality_ex=concatenate(inputSeasonality_exVect);

              firstE = season_exVect.begin() + i + 1;
              pastLastE = season_exVect.begin() + i + 1 + OUTPUT_SIZE;
              vector<Expression> outputSeasonality_exVect(firstE, pastLastE);  //[first,pastLast)
              outputSeasonality_ex = concatenate(outputSeasonality_exVect);

              input1_ex = cdiv(input1_ex, inputSeasonality_ex); // input deseasonalization
              labels1_ex = cdiv(labels1_ex, outputSeasonality_ex); //output deseasonalization
            }
            if (SEASONALITY_NUM > 1) {
              vector<Expression>::const_iterator firstE = season2_exVect.begin() + i + 1 - INPUT_SIZE;
              vector<Expression>::const_iterator pastLastE = season2_exVect.begin() + i + 1; //not including the last one
              vector<Expression> inputSeasonality2_exVect(firstE, pastLastE);  //[first,pastLast)
              inputSeasonality2_ex = concatenate(inputSeasonality2_exVect);

              firstE = season2_exVect.begin() + i + 1;
              pastLastE = season2_exVect.begin() + i + 1 + OUTPUT_SIZE;
              vector<Expression> outputSeasonality2_exVect(firstE, pastLastE);  //[first,pastLast)
              Expression outputSeasonality2_ex = concatenate(outputSeasonality2_exVect);

              input1_ex = cdiv(input1_ex, inputSeasonality2_ex); //input deseasonalization
              labels1_ex = cdiv(labels1_ex, outputSeasonality2_ex); //output deseasonalization
            }

            vector<Expression> joinedInput_ex;
            joinedInput_ex.emplace_back(noise(squash(cdiv(input1_ex, levels_exVect[i])), NOISE_STD)); //input normalization+noise
            joinedInput_ex.emplace_back(input(cg, { NUM_OF_CATEGORIES }, m4Obj.categories_vect));
            Expression input_ex = concatenate(joinedInput_ex);

            Expression labels_ex = squash(cdiv(labels1_ex, levels_exVect[i]));//output normalization

            Expression rnn_ex;
            try {
              rnn_ex = rNNStack[0].add_input(input_ex);
              for (int il=1; il<dilations.size(); il++)
                rnn_ex=rnn_ex+rNNStack[il].add_input(rnn_ex); //resNet-style
            }  catch (exception& e) {
              cerr<<"cought exception 2 while doing "<<series<<endl;
              cerr << e.what() << endl;
              cerr<<as_vector(input_ex.value())<<endl;
            }
            Expression out_ex;
            if (ADD_NL_LAYER) {
              out_ex=MLPW_ex*rnn_ex+MLPB_ex;
              out_ex = adapterW_ex*tanh(out_ex)+adapterB_ex;
            } else 
              out_ex=adapterW_ex*rnn_ex+adapterB_ex;

            Expression loss_ex=MSIS(out_ex, labels_ex);
            //Expression loss_ex = pinBallLoss(out_ex, labels_ex);
            if (i>=INPUT_SIZE+MIN_INP_SEQ_LEN)
                losses.push_back(loss_ex); 
          }//through points of a series

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
                cStateLosses_vEx.push_back(mean_elems(penalty_ex));
              }
          Expression cStateLossP_ex = average(cStateLosses_vEx)*C_STATE_PENALTY;
          cStateLoss = as_scalar(cStateLossP_ex.value());
          stateLosses.push_back(cStateLoss);
          loss_exp = loss_exp + cStateLossP_ex;
        }
          
        float loss = as_scalar(cg.forward(loss_exp));
        epochLosses.push_back(loss);//losses of all series in one epoch

        float forecastLoss = loss - levVarLoss - cStateLoss;
          forecLosses.push_back(forecastLoss);
        
          cg.backward(loss_exp);
          try {
            trainer->update();//update shared weights
            perSeriesTrainer->update();//update params of this series only
          } catch (exception& e) {//it may happen occasionally. I believe it is due to not robust enough implementation of squashing functions in Dynet. When abs(x)>35 NAs appear.
          //so the code below is trying to produce some diagnostics, hopefully useful when setting LEVEL_VARIABILITY_PENALTY and  C_STATE_PENALTY.
            cerr<<"cought exception while doing "<<series<<endl;
            cerr << e.what() << endl;
            
            float minSeason=BIG_FLOAT;
            for (int isea = 0; isea < season_exVect.size(); isea++) {
              float val= as_scalar(season_exVect[isea].value());
              //cout << " " << val;
              if (val<minSeason)
                minSeason=val;
            }  
            cout << "min season:"<<minSeason<<endl;

            minSeason = BIG_FLOAT;
            for (int isea = 0; isea < season2_exVect.size(); isea++) {
              float val = as_scalar(season2_exVect[isea].value());
              //cout << " " << val;
              if (val<minSeason)
                minSeason = val;
            }
            cout << "min season2:"<<minSeason<<endl;

            float minLevel = BIG_FLOAT;
            for (int isea = 0; isea < levels_exVect.size(); isea++) {
              float val = as_scalar(levels_exVect[isea].value());
              //cout << " " << val;
              if (val<minLevel)
                minLevel = val;
            }
            cout << "min level:"<<minLevel<<endl;

            float maxAbs = 0; int timeOfMax = 0; int layerOfMax = 0; int chunkOfMax=0;
            for (int irnn = 0; irnn < rNNStack.size(); irnn++) {
              auto state_vEx= rNNStack[irnn].c;//(time,layers)
              for (int it = 0; it < state_vEx.size(); it++) {  //through time
                for (int il = 0; il < state_vEx[it].size(); il++) {//through layers. Each layer has two states: c and h
                  auto state=as_vector(state_vEx[it][il].value());
                  for (int iv = 0; iv < state.size(); iv++) {
                    if (abs(state[iv]) > maxAbs) {
                      maxAbs = abs(state[iv]);
                      timeOfMax=it;
                      layerOfMax=il;
                      chunkOfMax= irnn;
                    }
                  }
                } //through layers/states
              } //through time
            }  //through chunks

            cout << "levSm:" << as_scalar(levSm_ex.value()) << endl;
            if (SEASONALITY_NUM > 0) 
              cout << "sSm:" << as_scalar(sSm_ex.value()) << endl;
            if (SEASONALITY_NUM > 1) 
              cout << "sSm2:" << as_scalar(sSm2_ex.value()) << endl;
            cout << "max abs:" << maxAbs <<" at time:"<< timeOfMax<<" at layer:"<< layerOfMax<<" and chunk:"<< chunkOfMax<<endl;

            //diagSeries.insert(series);
            pc.reset_gradient();
            perSeriesPC.reset_gradient();
          }

          //diagnostics saving
          AdditionalParamsF histAdditionalParams;
          histAdditionalParams.levSm=as_scalar(levSm_ex.value());
          if (iEpoch == 1 || iEpoch == NUM_OF_TRAIN_EPOCHS / 2 || iEpoch == NUM_OF_TRAIN_EPOCHS - 1) {
            for (int iv = 0; iv<levels_exVect.size(); iv++) {
              histAdditionalParams.levels.push_back(as_scalar(levels_exVect[iv].value()));
            }
          }

          if (SEASONALITY_NUM > 0) {
            histAdditionalParams.sSm=as_scalar(sSm_ex.value());
            for (int isea = 0; isea<SEASONALITY; isea++)
              histAdditionalParams.initSeasonality[isea] = as_scalar(season_exVect[isea].value());

            if (iEpoch == 1 || iEpoch == NUM_OF_TRAIN_EPOCHS / 2 || iEpoch == NUM_OF_TRAIN_EPOCHS - 1) {
              for (int iv = 0; iv<season_exVect.size(); iv++) {
                histAdditionalParams.seasons.push_back(as_scalar(season_exVect[iv].value()));
              }
            }
          }
         
          if (SEASONALITY_NUM > 1) {
            histAdditionalParams.sSm2 = as_scalar(sSm2_ex.value());
		        for (int isea=0; isea<SEASONALITY2; isea++) 
			        histAdditionalParams.initSeasonality2[isea]=as_scalar(season2_exVect[isea].value());   
               
            if (iEpoch == 1 || iEpoch == NUM_OF_TRAIN_EPOCHS / 2 || iEpoch == NUM_OF_TRAIN_EPOCHS - 1) {
              for (int iv = 0; iv<season2_exVect.size(); iv++) {
                histAdditionalParams.seasons2.push_back(as_scalar(season2_exVect[iv].value()));
              }
            }
          }     

          historyOfAdditionalParams_arr[iEpoch]=histAdditionalParams;
        }//through series

        float averageLoss = accumulate( epochLosses.begin(), epochLosses.end(), 0.0)/epochLosses.size();
        cout << ibig << " " << iEpoch << " " << inet << " count:" << oneNetAssignments.size() << " loss:" << averageLoss * 100;
        if (LEVEL_VARIABILITY_PENALTY > 0 || C_STATE_PENALTY > 0) {
          float averageForecLoss = accumulate(forecLosses.begin(), forecLosses.end(), 0.0) / forecLosses.size();
          cout << " forec loss:" << averageForecLoss * 100;
        }
        if (LEVEL_VARIABILITY_PENALTY > 0) {
          float averagelevVarLoss = accumulate(levVarLosses.begin(), levVarLosses.end(), 0.0) / levVarLosses.size();
          cout << " levVar loss:" << averagelevVarLoss * 100;
        }
        if (C_STATE_PENALTY > 0) {
          float averageStateLoss = accumulate(stateLosses.begin(), stateLosses.end(), 0.0) / stateLosses.size();
          cout << " state loss:" << averageStateLoss * 100;
        }
        cout<<endl;
      }//through nets. This should be done in parallel. One day it will, when Dynet allows it.


      //Validation. We just save outputs of all nets on all series
      //We can't attach validation to training, because training happens across subset of series*nets, and we need to store results from all of these combinations, for future use
      //level: epoch, but we do not use the epoch value, we overwrite
      for (int inet=0; inet<NUM_OF_NETS; inet++) { //through _all_ nets. Paralellize here.
        auto& rNNStack=rnnStack_arr[inet];
        Parameter& MLPW_par = MLPW_parArr[inet];
        Parameter& MLPB_par = MLPB_parArr[inet];
        Parameter& adapterW_par=adapterW_parArr[inet];
        Parameter& adapterB_par=adapterB_parArr[inet];

        for (auto iter = series_vect.begin() ; iter != series_vect.end(); ++iter) {//through _all_ series.
          string series=*iter;
          auto m4Obj=allSeries_map[series];

          ComputationGraph cg;
          for (int il=0; il<dilations.size(); il++) {
            rNNStack[il].new_graph(cg);
            rNNStack[il].start_new_sequence(); 
          }
          
          AdditionalParams& additionalParams=additionalParams_mapOfArr[series]->at(inet);
          Expression MLPW_ex, MLPB_ex;
          if (ADD_NL_LAYER) {
            MLPW_ex = parameter(cg, MLPW_par);
            MLPB_ex = parameter(cg, MLPB_par);
          }
          Expression adapterW_ex=parameter(cg, adapterW_par);
          Expression adapterB_ex=parameter(cg, adapterB_par);

          Expression levSmSerNet0_ex = parameter(cg, additionalParams.levSm);
          Expression levSm_ex = logistic(levSmSerNet0_ex);
          
          vector<Expression> season_exVect;//vector, because we do not know how long the series is
          Expression sSm_ex;
          if (SEASONALITY_NUM > 0) {
            Expression sSmSerNet0_ex= parameter(cg, additionalParams.sSm);
            sSm_ex = logistic(sSmSerNet0_ex);

            for (int isea = 0; isea<SEASONALITY; isea++) {
              Expression sSerNet0 = parameter(cg, additionalParams.initSeasonality[isea]);  //per series, per net
              Expression s1_ex = exp(sSerNet0);
              season_exVect.push_back(s1_ex);//Expression is a simple struct, without any storage management, so the auto copy constructor works OK.
            }
            season_exVect.push_back(season_exVect[0]);
          }

          vector<Expression> season2_exVect;//vector, because we do not know how long the series is
          Expression sSm2_ex;
          if (SEASONALITY_NUM > 1) {
            Expression sSm2SerNet0_ex= parameter(cg, additionalParams.sSm2);
            sSm2_ex = logistic(sSm2SerNet0_ex);

            for (int isea = 0; isea<SEASONALITY2; isea++) {
              Expression sSer2Net0 = parameter(cg, additionalParams.initSeasonality2[isea]);  //per series, per net
              Expression s2_ex = exp(sSer2Net0);
              season2_exVect.push_back(s2_ex);//Expression is a simple struct, without any storage management, so the auto copy constructor works OK.
            }
            season2_exVect.push_back(season2_exVect[0]);
          }

          vector<Expression> levels_exVect;
          if (SEASONALITY_NUM == 0) {
            levels_exVect.push_back(input(cg, m4Obj.vals[0]));
            for (int i = 1; i<m4Obj.vals.size(); i++) {
              Expression newLevel_ex = levSm_ex*m4Obj.vals[i] + (1 - levSm_ex)*levels_exVect[i - 1];
              levels_exVect.push_back(newLevel_ex);
            }
          }
          else if (SEASONALITY_NUM == 1) {
            Expression lev = cdiv(input(cg, m4Obj.vals[0]), season_exVect[0]);
            levels_exVect.push_back(lev);
            for (int i = 1; i<m4Obj.vals.size(); i++) {//if lback>0 then this is shortened, so it always contains data awe have right to access
              Expression newLevel_ex = m4Obj.vals[i] * cdiv(levSm_ex, season_exVect[i]) + (1 - levSm_ex)*levels_exVect[i - 1];
              levels_exVect.push_back(newLevel_ex);

              Expression newSeason_ex = m4Obj.vals[i] * cdiv(sSm_ex, newLevel_ex) + (1 - sSm_ex)*season_exVect[i];
              season_exVect.push_back(newSeason_ex);
            }

            //if prediction horizon is larger than seasonality, so we need to repeat some of the seasonality factors
            if (OUTPUT_SIZE>SEASONALITY) {
              unsigned long startSeasonalityIndx = season_exVect.size() - SEASONALITY;
              for (int i = 0; i<(OUTPUT_SIZE - SEASONALITY); i++)
                season_exVect.push_back(season_exVect[startSeasonalityIndx + i]);
            }
          }
          else if (SEASONALITY_NUM == 2) {
            Expression lev = cdiv(input(cg, m4Obj.vals[0]), season_exVect[0] * season2_exVect[0]);
            levels_exVect.push_back(lev);
            for (int i = 1; i<m4Obj.vals.size(); i++) {
              Expression newLevel_ex = m4Obj.vals[i] * cdiv(levSm_ex, season_exVect[i] * season2_exVect[i]) + (1 - levSm_ex)*levels_exVect[i - 1];
              levels_exVect.push_back(newLevel_ex);

              Expression newSeason_ex = m4Obj.vals[i] * cdiv(sSm_ex, newLevel_ex*season2_exVect[i]) + (1 - sSm_ex)*season_exVect[i];
              season_exVect.push_back(newSeason_ex);
              Expression newSeason2_ex = m4Obj.vals[i] * cdiv(sSm2_ex, newLevel_ex*season_exVect[i]) + (1 - sSm2_ex)*season2_exVect[i];
              season2_exVect.push_back(newSeason2_ex);
            }

            //if prediction horizon is larger than seasonality, so we need to repeat some of the seasonality factors
            if (OUTPUT_SIZE>SEASONALITY) {
              unsigned long startSeasonalityIndx = season_exVect.size() - SEASONALITY;
              for (int i = 0; i<(OUTPUT_SIZE - SEASONALITY); i++)
                season_exVect.push_back(season_exVect[startSeasonalityIndx + i]);
            }
            //if prediction horizon is larger than seasonality, so we need to repeat some of the seasonality factors
            if (OUTPUT_SIZE>SEASONALITY2) {
              unsigned long startSeasonalityIndx = season2_exVect.size() - SEASONALITY2;
              for (int i = 0; i<(OUTPUT_SIZE - SEASONALITY2); i++)
                season2_exVect.push_back(season2_exVect[startSeasonalityIndx + i]);
            }
          }
          else {
            cerr<<"SEASONALITY_NUM="<< SEASONALITY_NUM;
            exit(-1);
          }


          Expression inputSeasonality_ex; Expression inputSeasonality2_ex;
          Expression outputSeasonality_ex; Expression outputSeasonality2_ex;
          vector<Expression> losses;//losses of steps through single time series
          Expression out_ex;//we declare it here, bcause the last one will be the forecast
          for (int i=INPUT_SIZE-1; i<m4Obj.n; i++) {
            vector<float>::const_iterator first = m4Obj.vals.begin() + i + 1 - INPUT_SIZE;
            vector<float>::const_iterator pastLast = m4Obj.vals.begin() + i + 1; //not including the last one
            vector<float> input_vect(first, pastLast); //[first,pastLast)
            Expression input1_ex = input(cg, { INPUT_SIZE }, input_vect);

            if (SEASONALITY_NUM > 0 ) {
			        vector<Expression>::const_iterator firstE = season_exVect.begin() +i+1-INPUT_SIZE;
			        vector<Expression>::const_iterator pastLastE = season_exVect.begin() +i+1; //not including the last one
			        vector<Expression> inputSeasonality_exVect(firstE, pastLastE);  //[first,pastLast)
			        inputSeasonality_ex=concatenate(inputSeasonality_exVect);
              input1_ex = cdiv(input1_ex, inputSeasonality_ex); // input deseasonalization
            }
            if (SEASONALITY_NUM > 1) {
              vector<Expression>::const_iterator firstE = season2_exVect.begin() + i + 1 - INPUT_SIZE;
              vector<Expression>::const_iterator pastLastE = season2_exVect.begin() + i + 1; //not including the last one
              vector<Expression> inputSeasonality2_exVect(firstE, pastLastE);  //[first,pastLast)
              inputSeasonality2_ex = concatenate(inputSeasonality2_exVect);
              input1_ex = cdiv(input1_ex, inputSeasonality2_ex); //input deseasonalization
            }

            vector<Expression> joinedInput_ex;
            joinedInput_ex.emplace_back(noise(squash(cdiv(input1_ex, levels_exVect[i])), NOISE_STD)); //input normalization+noise
            joinedInput_ex.emplace_back(input(cg, { NUM_OF_CATEGORIES }, m4Obj.categories_vect));
            Expression input_ex = concatenate(joinedInput_ex);

            Expression rnn_ex;
            try {
              rnn_ex = rNNStack[0].add_input(input_ex);
              for (int il=1; il<dilations.size(); il++)
                rnn_ex=rnn_ex+rNNStack[il].add_input(rnn_ex);
            }  catch (exception& e) {
              cerr<<"cought exception 2 while doing "<<series<<endl;
              cerr << e.what() << endl;
              cerr<<as_vector(input_ex.value())<<endl;
            }
            if (ADD_NL_LAYER) {
              out_ex=MLPW_ex*rnn_ex+MLPB_ex;
              out_ex = adapterW_ex*tanh(out_ex)+adapterB_ex;
            } else 
              out_ex=adapterW_ex*rnn_ex+adapterB_ex;

            if (i<(m4Obj.n- OUTPUT_SIZE)) {//calc perf on training area
              vector<float>::const_iterator first = m4Obj.vals.begin() + i + 1;
              vector<float>::const_iterator pastLast = m4Obj.vals.begin() + i + 1 + OUTPUT_SIZE;
              vector<float> labels_vect(first, pastLast);  //[first,pastLast)
              Expression labels1_ex = input(cg, { OUTPUT_SIZE }, labels_vect);

              if (SEASONALITY_NUM > 0) {
                vector<Expression>::const_iterator firstE = season_exVect.begin() + i + 1;
                vector<Expression>::const_iterator pastLastE = season_exVect.begin() + i + 1 + OUTPUT_SIZE;
                vector<Expression> outputSeasonality_exVect(firstE, pastLastE);  //[first,pastLast)
                outputSeasonality_ex = concatenate(outputSeasonality_exVect);
                labels1_ex = cdiv(labels1_ex, outputSeasonality_ex); //output deseasonalization
              }
              if (SEASONALITY_NUM > 1) {
                vector<Expression>::const_iterator firstE = season2_exVect.begin() + i + 1;
                vector<Expression>::const_iterator pastLastE = season2_exVect.begin() + i + 1 + OUTPUT_SIZE;//checking if enough elements is in the vecor was done a few pe
                vector<Expression> outputSeasonality2_exVect(firstE, pastLastE);  //[first,pastLast)
                Expression outputSeasonality2_ex = concatenate(outputSeasonality2_exVect);
                labels1_ex = cdiv(labels1_ex, outputSeasonality2_ex); //output deseasonalization
              }
              Expression labels_ex = squash(cdiv(labels1_ex, levels_exVect[i]));//output normalization

          	  //Expression loss_ex = pinBallLoss(out_ex, labels_ex);
              Expression loss_ex = MSIS(out_ex, labels_ex);
          	  if (i>=INPUT_SIZE+MIN_INP_SEQ_LEN)
          			  losses.push_back(loss_ex);  //training area losses
            }
            
            if (i==(m4Obj.n-1)) {//validation loss
            	out_ex=expand(out_ex)*levels_exVect[i];//back to original scale
							if (SEASONALITY_NUM > 0 ) {
                vector<Expression>::const_iterator firstE = season_exVect.begin() + i + 1;
                vector<Expression>::const_iterator pastLastE = season_exVect.begin() + i + 1 + OUTPUT_SIZE;
                vector<Expression> outputSeasonality_exVect(firstE, pastLastE);  //[first,pastLast)
                for (int ios=0; ios<OUTPUT_SIZE; ios++) 
                  outputSeasonality_exVect.push_back(outputSeasonality_exVect[ios]);//we are duplicating it, as we deal with two outputs
                outputSeasonality_ex = concatenate(outputSeasonality_exVect);
                out_ex = cmult(out_ex, outputSeasonality_ex);//reseasonalize
              }
            	if (SEASONALITY_NUM > 1 ) {
                vector<Expression>::const_iterator firstE = season2_exVect.begin() + i + 1;
                vector<Expression>::const_iterator pastLastE = season2_exVect.begin() + i + 1 + OUTPUT_SIZE;
                vector<Expression> outputSeasonality2_exVect(firstE, pastLastE);  //[first,pastLast)
                for (int ios = 0; ios<OUTPUT_SIZE; ios++)
                  outputSeasonality2_exVect.push_back(outputSeasonality2_exVect[ios]);//we are duplicating it, as we deal with two outputs
                Expression outputSeasonality2_ex = concatenate(outputSeasonality2_exVect);
            		out_ex = cmult(out_ex, outputSeasonality2_ex);//reseasonalize
              }
                //we do not need the matching label here, because we do not bother calculate valid losses of each net across all series.
                //We care about best and topn performance
            }
          }//end of going through all point of a series
          
          Expression loss_exp = average(losses);
          float loss = as_scalar(cg.forward(loss_exp));//training loss of a single series
          netPerf_map[series][inet]=loss;
          
          //unordered_map<string, array<array<array<vector<float>, AVERAGING_LEVEL+1>, NUM_OF_NETS>, BIG_LOOP>> testResults_map((int)series_len*1.5);//per series, big loop, etc...
          //No epoch here, because this will just reflect the current (latest) situation - the last few epochs
          vector<float> out_vect=as_vector(out_ex.value());
          testResults_map[series][inet][iEpoch%AVERAGING_LEVEL]=out_vect;
          if (iEpoch>=AVERAGING_LEVEL && iEpoch % FREQ_OF_TEST==0) {
            vector<float> firstForec=testResults_map[series][inet][0];
            testResults_map[series][inet][AVERAGING_LEVEL]=firstForec;
            for (int ii=1; ii<AVERAGING_LEVEL; ii++) {
              vector<float> nextForec=testResults_map[series][inet][ii];
              for (int iii=0; iii<2*OUTPUT_SIZE; iii++)
                testResults_map[series][inet][AVERAGING_LEVEL][iii]+=nextForec[iii];
            }
            for (int iii=0; iii<2*OUTPUT_SIZE; iii++)
              testResults_map[series][inet][AVERAGING_LEVEL][iii]/=AVERAGING_LEVEL;
          } //time to average
        }//through series
      } //through nets
      
      if (iEpoch>0 && iEpoch % FREQ_OF_TEST==0) {
        //now that we have saved outputs of all nets on all series, let's calc how best and topn combinations performed during current epoch.
        vector<float> bestEpochLosses;
        vector<float> bestEpochAvgLosses;
        vector<float> topnEpochLosses;
        vector<float> topnEpochAvgLosses;
        vector<float> bestEpochLossesL;
        vector<float> bestEpochAvgLossesL;
        vector<float> topnEpochLossesL;
        vector<float> topnEpochAvgLossesL;
        vector<float> bestEpochLossesH;
        vector<float> bestEpochAvgLossesH;
        vector<float> topnEpochLossesH;
        vector<float> topnEpochAvgLossesH;
        
        for (auto iter = series_vect.begin() ; iter != series_vect.end(); ++iter) {
          string series=*iter;
          auto m4Obj=allSeries_map[series];

#if defined USE_ODBC        
          TRYODBC(hInsertStmt,
            SQL_HANDLE_STMT,
            SQLBindParameter(hInsertStmt, 4, SQL_PARAM_INPUT, SQL_C_CHAR, SQL_VARCHAR, 0, 0, (SQLCHAR*)series.c_str(), 0, &nullTerminatedStringOfSeries));

          TRYODBC(hInsertStmt,
            SQL_HANDLE_STMT,
            SQLBindParameter(hInsertStmt, OFFSET_TO_FIRST_ACTUAL + 2 * OUTPUT_SIZE + 3, SQL_PARAM_INPUT, SQL_C_SLONG, SQL_INTEGER, 0, 0, (SQLPOINTER)&m4Obj.n, 0, NULL));
#endif 
          
          float avgLoss;
          vector<float> avgLatest;
          vector<float> avgAvg;
          
          for (int itop=0; itop<TOPN; itop++) {
            int inet=netRanking_map[series][itop];
            
            if (itop==0) {
              if (LBACK > 0) {
                float qLoss = errorFunc(testResults_map[series][inet][iEpoch%AVERAGING_LEVEL], m4Obj.testVals, m4Obj.meanAbsSeasDiff);
                bestEpochLosses.push_back(qLoss);

                qLoss=wQuantLoss(testResults_map[series][inet][iEpoch%AVERAGING_LEVEL], m4Obj.testVals, TAUL, 0);
                bestEpochLossesL.push_back(qLoss);

                qLoss = wQuantLoss(testResults_map[series][inet][iEpoch%AVERAGING_LEVEL], m4Obj.testVals, TAUH, OUTPUT_SIZE);
                bestEpochLossesH.push_back(qLoss);
              }
              avgLatest=testResults_map[series][inet][iEpoch%AVERAGING_LEVEL];  //used later for calculating topn loss
              
              if (iEpoch>=AVERAGING_LEVEL) {
                if (LBACK > 0) {
                  float qLoss = errorFunc(testResults_map[series][inet][AVERAGING_LEVEL], m4Obj.testVals, m4Obj.meanAbsSeasDiff);
                  bestEpochAvgLosses.push_back(qLoss);

                  qLoss = wQuantLoss(testResults_map[series][inet][AVERAGING_LEVEL], m4Obj.testVals, TAUL, 0);
                  bestEpochAvgLossesL.push_back(qLoss);

                  qLoss = wQuantLoss(testResults_map[series][inet][AVERAGING_LEVEL], m4Obj.testVals, TAUH, OUTPUT_SIZE);
                  bestEpochAvgLossesH.push_back(qLoss);
                }
                avgAvg=testResults_map[series][inet][AVERAGING_LEVEL];
              }
            } else {
              for (int iii=0; iii<2*OUTPUT_SIZE; iii++) {
                avgLatest[iii]+=testResults_map[series][inet][iEpoch%AVERAGING_LEVEL][iii];//calculate current topn
                if (iEpoch>=AVERAGING_LEVEL)
                  avgAvg[iii]+=testResults_map[series][inet][AVERAGING_LEVEL][iii];
              }
            }
          }//through topn
          
          for (int iii=0; iii<2*OUTPUT_SIZE; iii++)
	          avgLatest[iii]/=TOPN;

          if (LBACK > 0) {
            float qLoss = errorFunc(avgLatest, m4Obj.testVals, m4Obj.meanAbsSeasDiff);
            topnEpochLosses.push_back(qLoss);

            qLoss = wQuantLoss(avgLatest, m4Obj.testVals, TAUL, 0);
            topnEpochLossesL.push_back(qLoss);

            qLoss = wQuantLoss(avgLatest, m4Obj.testVals, TAUH, OUTPUT_SIZE);
            topnEpochLossesH.push_back(qLoss);
          }
          
          if (iEpoch>=AVERAGING_LEVEL) {
            for (int iii = 0; iii<2*OUTPUT_SIZE; iii++) 
              avgAvg[iii] /= TOPN;

            finalResults_map[series] = avgAvg;

            if (LBACK > 0) {
#if defined USE_ODBC        
              TRYODBC(hInsertStmt,
                SQL_HANDLE_STMT,
                SQLBindParameter(hInsertStmt, OFFSET_TO_FIRST_ACTUAL + 2 * OUTPUT_SIZE + 1, SQL_PARAM_INPUT, SQL_C_FLOAT, SQL_FLOAT, 0, 0, (SQLPOINTER)&avgLoss, 0, NULL));
          
              for (int iv=0; iv<2; iv++)  {
                if (iv==0)
                  TRYODBC(hInsertStmt,
                    SQL_HANDLE_STMT,
                    SQLBindParameter(hInsertStmt, 1, SQL_PARAM_INPUT, SQL_C_CHAR, SQL_VARCHAR, 0, 0, (SQLCHAR*)runL.c_str(), 0, &nullTerminatedStringOfRun))
                else                                                      
                  TRYODBC(hInsertStmt,
                    SQL_HANDLE_STMT,
                    SQLBindParameter(hInsertStmt, 1, SQL_PARAM_INPUT, SQL_C_CHAR, SQL_VARCHAR, 0, 0, (SQLCHAR*)runH.c_str(), 0, &nullTerminatedStringOfRun));
                
                for (int iii=0; iii<OUTPUT_SIZE; iii++) {              
                  int ipos=OFFSET_TO_FIRST_ACTUAL + 1 + 2*iii;
                  TRYODBC(hInsertStmt,
                      SQL_HANDLE_STMT,
                      SQLBindParameter(hInsertStmt, ipos, SQL_PARAM_INPUT, SQL_C_FLOAT, SQL_FLOAT, 0, 0, (SQLPOINTER)&m4Obj.testVals[iii], 0, NULL));

                  TRYODBC(hInsertStmt,
                      SQL_HANDLE_STMT,
                      SQLBindParameter(hInsertStmt, ipos+1, SQL_PARAM_INPUT, SQL_C_FLOAT, SQL_FLOAT, 0, 0, (SQLPOINTER)&avgAvg[iii+iv*OUTPUT_SIZE], 0, NULL));
                }
                TRYODBC(hInsertStmt,
                 SQL_HANDLE_STMT,
                 SQLExecute(hInsertStmt));                 
              }
#endif               
              float qLoss = errorFunc(avgAvg, m4Obj.testVals, m4Obj.meanAbsSeasDiff);
              topnEpochAvgLosses.push_back(qLoss);

              qLoss = wQuantLoss(avgAvg, m4Obj.testVals, TAUL, 0);
              topnEpochAvgLossesL.push_back(qLoss);

              qLoss = wQuantLoss(avgAvg, m4Obj.testVals, TAUH, OUTPUT_SIZE);
              topnEpochAvgLossesH.push_back(qLoss);
            }
          }
        }//through series
        if (LBACK > 0) {
          float bestEpochLoss=accumulate( bestEpochLosses.begin(), bestEpochLosses.end(), 0.0)/bestEpochLosses.size();
          float topnEpochLoss=accumulate( topnEpochLosses.begin(), topnEpochLosses.end(), 0.0)/topnEpochLosses.size();
          float bestEpochLossL = accumulate(bestEpochLossesL.begin(), bestEpochLossesL.end(), 0.0) / bestEpochLossesL.size();
          float topnEpochLossL = accumulate(topnEpochLossesL.begin(), topnEpochLossesL.end(), 0.0) / topnEpochLossesL.size();
          float bestEpochLossH = accumulate(bestEpochLossesH.begin(), bestEpochLossesH.end(), 0.0) / bestEpochLossesH.size();
          float topnEpochLossH = accumulate(topnEpochLossesH.begin(), topnEpochLossesH.end(), 0.0) / topnEpochLossesH.size();
          cout<<ibig<<" "<<iEpoch<<" VALID best:"<<bestEpochLoss<<" L:"<< bestEpochLossL<<" H:"<< bestEpochLossH<<
            " topn:"<<topnEpochLoss<<" L:"<< topnEpochLossL<<" H:"<< topnEpochLossH;
          if (iEpoch>=AVERAGING_LEVEL) {
            float bestEpochAvgLoss=accumulate( bestEpochAvgLosses.begin(), bestEpochAvgLosses.end(), 0.0)/bestEpochAvgLosses.size();
            float topnEpochAvgLoss=accumulate( topnEpochAvgLosses.begin(), topnEpochAvgLosses.end(), 0.0)/topnEpochAvgLosses.size();
            float bestEpochAvgLossL = accumulate(bestEpochAvgLossesL.begin(), bestEpochAvgLossesL.end(), 0.0) / bestEpochAvgLossesL.size();
            float topnEpochAvgLossL = accumulate(topnEpochAvgLossesL.begin(), topnEpochAvgLossesL.end(), 0.0) / topnEpochAvgLossesL.size();
            float bestEpochAvgLossH = accumulate(bestEpochAvgLossesH.begin(), bestEpochAvgLossesH.end(), 0.0) / bestEpochAvgLossesH.size();
            float topnEpochAvgLossH = accumulate(topnEpochAvgLossesH.begin(), topnEpochAvgLossesH.end(), 0.0) / topnEpochAvgLossesH.size();
            cout<<" bestAvg:"<<bestEpochAvgLoss<<" L:"<< bestEpochAvgLossL<<" H:"<< bestEpochAvgLossH<<
              " topnAvg:"<<topnEpochAvgLoss<<" L:"<< bestEpochAvgLossL<<" H:"<< bestEpochAvgLossH<<endl;
          } else
            cout<<endl;
        }
      }//time to report
      
      //assign
      for (int inet=0; inet<NUM_OF_NETS; inet++)
        seriesAssignment[inet].clear();
      for (auto iter = series_vect.begin() ; iter != series_vect.end(); ++iter) {
        string series=*iter;
        //unordered_map<string, array<int, NUM_OF_NETS>> netRanking_map
        netRanking_map[series]=perfToRanking(netPerf_map[series]);
        
        for (int itop=0; itop<TOPN; itop++) {
          int inet=netRanking_map[series][itop];
          seriesAssignment[inet].push_back(series); //every net has a set
        }
      }
      
      //check and fix degenerations
      for (int inet=0; inet<NUM_OF_NETS; inet++) {
        if (seriesAssignment[inet].size()==0) {
          cout<<"Resetting "<<inet<<endl;
          for (int i=0; i<series_len/2; i++) {
            int irand=uniOnSeries(rng);
            seriesAssignment[inet].push_back(series_vect[irand]);
          }
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
    }//through epochs of RNN
    
    //some diagnostic info
    set<string> diagSeries;
    for (int i=0; i<1; i++) {//add a few normal ones
      int irand=uniOnSeries(rng);
      diagSeries.insert(series_vect[irand]);
    }
    for(auto series : diagSeries) {
      cout<<endl<<series<<endl;
      
      cout<<"lSm:"<<endl;
      for (int inet=0; inet<NUM_OF_NETS; inet++) {
        cout<<"inet:"<<inet<<" ";
    	auto& historyOfAdditionalParams_arr=historyOfAdditionalParams_map[series]->at(inet);
        for (int iEpoch=0; iEpoch<NUM_OF_TRAIN_EPOCHS; iEpoch++)
            cout<<historyOfAdditionalParams_arr[iEpoch].levSm<<" ";
        cout<<endl;
      }
      
      if (SEASONALITY_NUM > 0 ) {
        cout<<"sSm:"<<endl;
        for (int inet=0; inet<NUM_OF_NETS; inet++) {
          cout<<"inet:"<<inet<<" ";
    	    auto& historyOfAdditionalParams_arr=historyOfAdditionalParams_map[series]->at(inet);
          for (int iEpoch=0; iEpoch<NUM_OF_TRAIN_EPOCHS; iEpoch++)
            cout<<historyOfAdditionalParams_arr[iEpoch].sSm<<" ";
          cout<<endl;
        }
      }  
      
      if (SEASONALITY_NUM > 1 ) {
        cout<<"sSm2:"<<endl;
        for (int inet=0; inet<NUM_OF_NETS; inet++) {
          cout<<"inet:"<<inet<<" ";
    	  auto& historyOfAdditionalParams_arr=historyOfAdditionalParams_map[series]->at(inet);
          for (int iEpoch=0; iEpoch<NUM_OF_TRAIN_EPOCHS; iEpoch++)
            cout<<historyOfAdditionalParams_arr[iEpoch].sSm2<<" ";
        cout<<endl;
        }
      }
      
      for (int inet = 0; inet<NUM_OF_NETS; inet++) {
        cout<<"inet:"<<inet<<" ";
        auto& historyOfAdditionalParams_arr = historyOfAdditionalParams_map[series]->at(inet);
        for (int iEpoch = 0; iEpoch<NUM_OF_TRAIN_EPOCHS; iEpoch++) {
          if (historyOfAdditionalParams_arr[iEpoch].levels.size()>0) {
            cout << "levels:" << iEpoch<<" ";
            for (int iv = 0; iv<historyOfAdditionalParams_arr[iEpoch].levels.size(); iv++)
              cout << historyOfAdditionalParams_arr[iEpoch].levels[iv] << ", ";
            cout << endl;
            if (SEASONALITY_NUM > 0 ) {
              cout << "seasons:" << iEpoch<<" ";
              for (int iv = 0; iv<historyOfAdditionalParams_arr[iEpoch].levels.size(); iv++)
                cout << historyOfAdditionalParams_arr[iEpoch].seasons[iv] << ", ";
              cout << endl;
            }
            if (SEASONALITY_NUM > 1 ) {
              cout << "seasons2:" << iEpoch<<" ";
              for (int iv = 0; iv<historyOfAdditionalParams_arr[iEpoch].levels.size(); iv++)
                cout << historyOfAdditionalParams_arr[iEpoch].seasons2[iv] << ", ";
              cout << endl;
            }
          }
        }
      }
    }//end of diag printing
    
    //save the forecast to outputFile
    ofstream outputFile;
    outputFile.open(outputPathL);
    for (auto iter = series_vect.begin(); iter != series_vect.end(); ++iter) {
      string series = *iter;
      outputFile<< series;
      for (int io=0; io<OUTPUT_SIZE; io++)
        outputFile << ", " << finalResults_map[series][io];
      outputFile<<endl;
    }
    outputFile.close();
    
    outputFile.open(outputPathH);
    for (auto iter = series_vect.begin(); iter != series_vect.end(); ++iter) {
      string series = *iter;
      outputFile << series;
      for (int io = 0; io<OUTPUT_SIZE; io++)
        outputFile << ", " << finalResults_map[series][io+OUTPUT_SIZE];
      outputFile << endl;
    }
    outputFile.close();
    
    //delete    
    for (int inet = 0; inet<NUM_OF_NETS; inet++) {
      delete trainers_arr[inet];
      perSeriesTrainers_arr[inet];
    }

    for (auto iter = series_vect.begin() ; iter != series_vect.end(); ++iter) {
      string series=*iter;
      delete additionalParams_mapOfArr[series];
      delete historyOfAdditionalParams_map[series];
    }
    additionalParams_mapOfArr.clear();
    historyOfAdditionalParams_map.clear();
  }//big loop
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
