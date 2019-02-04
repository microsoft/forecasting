# Merging outputs, per category, M4 competition, for Prediction Intervals , so for ES_RNN_PI and ES_RNN_E_PI
# Author: Slawek Smyl, Mar-May 2018


#The c++ executables write to one (occasinally two, sorry :-), so in such case move files to one dir before continuing) directories. 
#(One logical run of several instances of the same program will produce a number files, e.g. outputs with different ibig value) 
#This script merges, averages values, and writes them down to the same directory - FOREC_DIR
###############################################################################

#directory that should include all *-train.csv files, as well as M4-info.csv
DATA_DIR="F:/progs/data/M4DataSet/"
m4Info_df=read.csv(paste0(DATA_DIR,"M4-info.csv"))
options(stringsAsFactors =FALSE)
memory.limit(10000)

#directory with all the output files produced by the c++ code  we want to merge
FOREC_DIR='F:\\progs\\data\\M4\\Hourlygood'  #do not end with separator

LBACK=1  #shoud be as in the c++ code, LBACK>0 means backtesting
#SP="Quarterly"
#SP="Yearly"
#SP="Daily"
SP="Hourly"
m4_df=read.csv(paste0(DATA_DIR,SP,"-train.csv"))


#//----------------PARAMS ----------   comment/uncomment following 3 variables
#for ES_RNN_E_PI, so for all except Monthly and Quarterly runs: 
NUM_OF_SEEDS=1
NUM_OF_CHUNKS=1
#IBIGS=<number of input files n FOREC_DIR>/2
IBIGS=6

#for ES_RNN_PI (do for Monthly and Quarterly):
#NUM_OF_CHUNKS=2 #same as NUM_OF_CHUNKS constant the the c++ cource code, changing it is not recommended. 
#NUM_OF_SEEDS=3  #It is equal to the number of seeds in the startup script, (or number of teams of worker processes)
# so number_of_concurrent_executables==number_of_lines_in_the_running script/NUM_OF_CHUNKS, and number_of_chunks 
#E.g if using following script for ES_RNN:
# start <this_executable> 10 1 0
# start <this_executable> 10 2 0
# start <this_executable> 20 1 5
# start <this_executable> 20 2 5
# start <this_executable> 30 1 10
# start <this_executable> 30 2 10
# we have here three seeds: 10,20,30, and two chunks: 1,2. (The pairs of workes have IBIG offsets of 0,5,10)
#IBIGS=3 #number of complete runs by each executables, so if programs are not interrupted, this should be equal to the constant BIG_LOOP in the c++ code, by default 3.

ALPHA = 0.05;
ALPHA_MULTIP = 2 / ALPHA;

MSIS<-function(forecL,forecH,actual) {
	sumDiffs=0
	for (i in 1:(length(actual)-seasonality)) {
		sumDiffs=sumDiffs+abs(actual[i+seasonality]-actual[i])
	}
	avgAbsDiff=sumDiffs/(length(actual)-seasonality)
	
	actual=actual[(length(actual)-LBACK*horizon+1):(length(actual)-(LBACK-1)*horizon)]
	
	msis=sum(forecH-forecL)+sum(pmax(0,forecL-actual))*ALPHA_MULTIP+sum(pmax(0,actual-forecH))*ALPHA_MULTIP
	msis/horizon/avgAbsDiff
}
errorFunc=MSIS

spInfo_df=m4Info_df[m4Info_df$SP==SP,]
ids=spInfo_df$M4id
horizon=spInfo_df[1,"Horizon"]
seasonality=spInfo_df[1,"Frequency"]


#lower
#VARIABLE + "_" + to_string(seedForChunks) + "_" + to_string(chunkNo) + "_" + to_string(ibigDb)+"_LB"+ to_string(LBACK)+ ".csv";
inputFiles=list.files(path = FOREC_DIR, pattern = paste0(SP,".*LLB",LBACK), full.names = T)
if (length(inputFiles)!=NUM_OF_SEEDS*NUM_OF_CHUNKS*IBIGS) {
	stop("length(inputFiles)!=NUM_OF_SEEDS*NUM_OF_CHUNKS*IBIGS")
}

comp_df=NULL
fil=inputFiles[1]
for (fil in inputFiles) {
  print(fil)
	c_df=read.csv(fil, header=F)
	comp_df=rbind(comp_df,c_df)
} 
names(comp_df)[1]='id'

forecSeries=sort(unique(comp_df$id))
if (length(forecSeries)!=length(ids) && LBACK==0) {
	stop(paste0("Expected number of cases:",length(ids)," but got:",length(forecSeries)))
}

SIZE_OF_CHUNK=1000
out_df=NULL; ou_df=NULL
fSeries=forecSeries[1]
for (fSeries in forecSeries) {
	oneSeriesForecs_df=comp_df[comp_df$id==fSeries,]
	o1=colMeans(oneSeriesForecs_df[,2:ncol(oneSeriesForecs_df)])
	o_df=data.frame(id=fSeries, as.list(o1), stringsAsFactors =F)
	ou_df=rbind(ou_df, o_df)
	if (nrow(ou_df)>=SIZE_OF_CHUNK) {
		out_df=rbind(out_df,ou_df)
		ou_df=NULL
		print(nrow(out_df))
	}
}
out_df=rbind(out_df,ou_df)
print(nrow(out_df))
out_df=out_df[order(as.integer(substring(out_df$id, 2))),]

outPath=paste0(FOREC_DIR,'\\',SP,"ForecL.csv")
write.csv(out_df,file=outPath,row.names = F)

lower_df=out_df

#####################################
#higher
inputFiles=list.files(path = FOREC_DIR, pattern = paste0(SP,".*HLB",LBACK), full.names = T)
if (length(inputFiles)!=NUM_OF_SEEDS*NUM_OF_CHUNKS*IBIGS) {
	stop("length(inputFiles)!=NUM_OF_SEEDS*NUM_OF_CHUNKS*IBIGS")
}

comp_df=NULL
fil=inputFiles[1]
for (fil in inputFiles) {
	print(fil)
	c_df=read.csv(fil, header=F)
	comp_df=rbind(comp_df,c_df)
} 
names(comp_df)[1]='id'

forecSeries=sort(unique(comp_df$id))
if (length(forecSeries)!=length(ids) && LBACK==0) {
	print(paste0("Warning. Expected number of cases:",length(ids)," but got:",length(forecSeries)))
}

SIZE_OF_CHUNK=1000
out_df=NULL; ou_df=NULL
fSeries=forecSeries[1]
for (fSeries in forecSeries) {
	oneSeriesForecs_df=comp_df[comp_df$id==fSeries,]
	o1=colMeans(oneSeriesForecs_df[,2:ncol(oneSeriesForecs_df)])
	o_df=data.frame(id=fSeries, as.list(o1), stringsAsFactors =F)
	ou_df=rbind(ou_df, o_df)
	if (nrow(ou_df)>=SIZE_OF_CHUNK) {
		out_df=rbind(out_df,ou_df)
		ou_df=NULL
		print(nrow(out_df))
	}
}
out_df=rbind(out_df,ou_df)
print(nrow(out_df))
out_df=out_df[order(as.integer(substring(out_df$id, 2))),]

outPath=paste0(FOREC_DIR,'\\',SP,"ForecH.csv")
write.csv(out_df,file=outPath,row.names = F)

higher_df=out_df


################ Main work done, now just diagnostics calculations and plots

#display a sample of forecasts and, if LBACK>0,  actuals
MAX_NUM_OF_POINTS_TO_SHOW=200
i=1
for (i in 1:100) {
	irand=sample(1:length(forecSeries),1)
	fSeries=forecSeries[irand]
	forecL=as.numeric(lower_df[lower_df$id==fSeries,2:ncol(lower_df)])
	forecH=as.numeric(higher_df[higher_df$id==fSeries,2:ncol(higher_df)])
	actual=as.numeric(m4_df[m4_df$V1==fSeries,2:ncol(m4_df)])
	actual=actual[!is.na(actual)]
	if (length(actual)>MAX_NUM_OF_POINTS_TO_SHOW) {
		actual=actual[(length(actual)-MAX_NUM_OF_POINTS_TO_SHOW):length(actual)]	
	}
	if (LBACK==0) {
		plot(c(actual,forecH), col=c(rep(1,length(actual)),rep(2,length(forecH))), main=fSeries)
    lines(c(actual,forecL), col=c(rep(1,length(actual)),rep(3,length(forecL))), type='p')		
	} else {
		ymin=min(actual,forecL)
		ymax=max(actual,forecH)
		plot(1:length(actual),actual, main=fSeries, ylim=c(ymin,ymax))
		lines((length(actual)-length(forecH)+1):length(actual), forecH, col=2, type='p')
		lines((length(actual)-length(forecL)+1):length(actual), forecL, col=3, type='p')
	}
	
	Sys.sleep(5)
}



#calc error metric: MSIS
if (LBACK>0) {
	summErrors=0
	fSeries=forecSeries[1]
	i=1
	for (fSeries in forecSeries) {
		if (i%%1000==0)
			cat(".")
		forecL=as.numeric(lower_df[lower_df$id==fSeries,2:ncol(lower_df)])
		forecH=as.numeric(higher_df[higher_df$id==fSeries,2:ncol(higher_df)])
		actual=as.numeric(m4_df[m4_df$V1==fSeries,2:ncol(m4_df)])
		actual=actual[!is.na(actual)]
		summErrors=summErrors+errorFunc(forecL, forecH, actual)
		i=i+1
	}
	print(".")
	print(paste0("avg error:",round(summErrors/length(forecSeries),2)))
}


