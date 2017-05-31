import numpy as np
import itertools
from collections import defaultdict
from datetime import datetime
import math
import sys
import os
from scipy.stats import percentileofscore
import graphlab as gl
import graphlab.aggregate as agg
gl.set_runtime_config('GRAPHLAB_CACHE_FILE_LOCATIONS','/home/mraza/tmp/')
gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 48)

# X1,X2,X3,X4,X5,X6,X7
# 2015-10-01 12:08:41,1046885725705,1046910448494,GSM,ITN006,,1.5
# 2015-10-01 16:55:32,1046885725705,1046910448494,GSM,ITN010,,1.5


def distance(l1_lat,l1_lng,l2_lat,l2_lng):
    R = 6371; # Radius of the earth in km
    d=0.0
    try:
            l1_lat, l1_lng, l2_lat, l2_lng=float(l1_lat), float(l1_lng), float(l2_lat), float(l2_lng)
    except:
            l1_lat, l1_lng, l2_lat, l2_lng=0.0,0.0,0.0,0.0
    dLat = (l1_lat-l2_lat)*math.pi/180
    dLon = (l1_lng-l2_lng)*math.pi/180
    a = math.sin(dLat/2) * math.sin(dLat/2) +math.cos((l1_lat)*math.pi/180) * math.cos((l2_lat)*math.pi/180) * math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c # Distance in km
    return d


def _calc_rog(seq):
    lat_lst,lng_lst=[],[]
    for s in seq:
        if s in loc_dct:
            lat1, lng1=loc_dct[s]
            lat_lst.append(lat1)
            lng_lst.append(lng1)
    centroid_lat=np.nanmean(lat_lst)
    centroid_lng=np.nanmean(lng_lst)
    cdistance=0.0
    for i in range(len(lat_lst)):
        cdistance+=distance(lat_lst[i], lng_lst[i], centroid_lat, centroid_lng)
    if len(lat_lst)!=0:
        cdistance/=len(lat_lst)
    else:
        cdistance=-1.0
    return cdistance


def _diversity(seq):
    #print 'len(seq)', len(seq)
    # print seq
    dc=defaultdict(int)
    for each in seq:
        dc[each]+=1
    unique_count=len(dc.keys())
    total_comm=float(np.sum(dc.values()))
    for each in dc:
        dc[each]=dc[each]/total_comm
    numerator=np.sum([dc[each]*math.log(dc[each] )for each in dc])
    denominator=-math.log(unique_count)
    if unique_count!=1:
        ret= float(numerator/denominator)
        #print ret
        return ret
    else:
        return -1.0


def _avg_distance(x,y):
    len_x1, len_y1=len(x), len(y)
    if len_x1==0 or len_y1==0:
        return -1
    else:
        zipped=zip(x,y)
        avg_distance=0
        present=-1
        count=0
        for x1,y1 in zipped:
            if x1=='' or y1=='':
                avg_distance+=0
            else:
                if x1 in loc_dct and y1 in loc_dct:
                    lat1,lng1=loc_dct[x1]
                    lat2,lng2=loc_dct[y1]
                    avg_distance+=distance(lat1, lng1, lat2, lng2)
                    count+=1
                present=1
    if (present==0 and avg_distance==0) or count==0: # None of the tuples had both caller cell and the receiver cell
        return -1
    else:
        return avg_distance/count


def _avgdiff(seq):
    lst=[]
    for d1 in seq:
        #print d1
        lst.append(d1)#datetime.strptime(d1, "%Y-%m-%d %H:%M:%S"))
    #print 'Type {} {}'.format( lst, type(lst[0]))
    lst2=[ (lst[i]- lst[i+1]).seconds for i in range(len(lst)-1)]
    if len(lst2)==0:
        return 0.0
    else:
        return float(np.nanmean(lst2))


def _unique_count(x):
    return len(np.unique(x))


def _rank(x,y):
    # print x, y
    ret= percentileofscore(x,y)
    return ret
    # print ret, type(ret)
    # if type(ret)!=float:
    #   return -1
    # else:
    #   return ret





def getSeedDf(filename):
    seeds_df=gl.SFrame.read_csv(filename)
    return seeds_df


def castDate(x):
    try:
        y=datetime.strptime(x,"%Y%m%dT%H%M%S")
    except:
        try:
            y=datetime.strptime(x,"%Y-%m-%d %H:%M")
        except:
            y=datetime.strptime("2017-02-28","%Y-%m-%d")
            pass
        pass
    return y

def processDate(df):    
    df['Date2']=df['Date'].apply(lambda x:castDate(x))
    df['Day']=df['Date2'].apply(lambda x:str(x.year)+'-'+str(x.month)+'-'+str(x.day))
    df['IsWeekend']=df['Date2'].apply(lambda x:x.weekday() in [5,6])
    df=df.remove_column('Date')
    df.rename({'Date2':'Date'})

def getDf(filename, header, split_datetime=False):
    df=gl.SFrame.read_csv(filename)
    df=df.rename(header)
    if split_datetime:
        df['Date2']=df['Date'].apply(lambda x:castDate(x))
        df['Day']=df['Date2'].apply(lambda x:str(x.year)+'-'+str(x.month)+'-'+str(x.day))
        df['IsWeekend']=df['Date2'].apply(lambda x:x.weekday() in [5,6])
        df=df.remove_column('Date')
        df.rename({'Date2':'Date'})
    df['Direction']='Out'
    return df


def load_locations(loc_filename='./telenor_sites3.csv'):
    dc={}
    with open(loc_filename) as fin:
        fin.next()# skip the header
        for each in fin:
            tower, lat, lng=each.split(',')
            lat=float(lat)
            lng=float(lng)
            dc[tower]=(lat,lng)
    return dc


def generate_pivots(cols_list):
    keyCol='CallerId'
    comb_params=[[keyCol]] # list of lists
    exclude=[keyCol,'Duration','Date2','Date','Type','alter','GeoDistance']
    #cols_list=voice_df2.column_names()
    cols_list2=[each for each in cols_list if each not in exclude and each not in partition_columns]
    for i in xrange(1, len(cols_list2)+1):
        els = [[keyCol]+list(x) for x in itertools.combinations(cols_list2, i)]
        comb_params.extend(els)
    comb_params
    return comb_params


# df_ret=apply_aggregate(voice_df2, comb, operations.keys(),operations , postfix)
def apply_aggregate(df2,pivot, columns, operations, postfix):
    groupBy=pivot
    cols=set(columns)-set(pivot)
    #funcs=[count, mean, stddev]
    #exprs = [f(col(c)).alias(operations_names_dict[f]+'('+c+')'+'_Per_'+postfix) for c in cols for f in operations[c] ]
    exprs={}
    for c in cols:
        for f in operations[c]:
            exprs[operations_names_dict[f]+'('+c+')'+'_Per_'+postfix]=f(c)
    #print exprs
    df_ret=df2.groupby(key_columns=groupBy, operations=exprs)
    #df_ret.withColumn('avgdiff',avgdiff2(df_ret['joincollect(Date)']))
    return df_ret



def align_for_network_aggregate(features_df, raw_df):
    raw_df=raw_df[["ReceiverId"]]#,"CallerId"]]#.rename({'CallerId':'JoinKey'})
    #features_df=features_df.rename({'CallerId':'JoinKey'})
    joined_df=features_df.join(raw_df, {'CallerId':'ReceiverId'}, 'left')
    #joined_df=joined_df.rename({'JoinKey':'CallerId'})
    if 'ReceiverId' in joined_df.column_names():
        joined_df=joined_df.remove_column('ReceiverId')
    return joined_df


def network_aggregate(joined_df, postfix):
    groupBy=['CallerId']
    cols=set(joined_df.column_names())-set(['CallerId','ReceiverId'])
    funcs=[ agg.MEAN, agg.STD,agg.SUM]# removing count as it would be equal to the degree
    exprs={}
    for c in cols:
        for f in funcs:
            exprs[operations_names_dict[f]+'('+c+')'+'_Per_'+postfix]=f(c)
    df_ret=joined_df.groupby(key_columns=groupBy, operations=exprs)
    return df_ret


def remove_column(df_ret, key):
    for c in df_ret.column_names():
        if key in c:
            #print 'Dropping', c
            df_ret=df_ret.remove_column(c)
    return df_ret


def apply_special_operations(special_operations, pivot,df_ret, postfix):
    final_list=list(set(special_operations)-set(pivot))
    print 'final_list',final_list
    print 'pivot',pivot
    for key in final_list:
        for op in special_operations[key]:
            #print 'key={}, op={}'.format(key, str(op))
            if type(key)!=tuple:
                df_ret[operations_names_dict[op]+'('+key+')'+'_Per_'+postfix]=df_ret['joincollect('+key+')'+'_Per_'+postfix].apply(lambda x:op(x))
            elif len(key)==2 and key[0] in final_list and key[1] in final_list:
                df_ret[operations_names_dict[op]+'('+key[0]+';'+key[1]+')'+'_Per_'+postfix]=df_ret.apply(lambda x: op(x['joincollect('+key[0]+')'+'_Per_'+postfix],
                    x['joincollect('+key[1]+')'+'_Per_'+postfix]))
    return df_ret


def composite_reduce(df_ret, comb):
    operations2={}
    new_cols=set(df_ret.column_names())-set(comb)
    for each in new_cols:
        operations2[each]=[agg.STD]
        # removing mean as mean of mean is equal to the original mean, same for count, min , max, sum
    df_ret_base=apply_aggregate(df_ret,['CallerId'], new_cols, operations2, postfix='CallerId') 
    #print 'Reduce complete'
    return df_ret_base



def merge_dfs_from_dict(features_dc):
    lst=[k  for k in features_dc.keys() if 'Date' not in k]
    final_merged_df=features_dc[lst[0]]
    print final_merged_df.shape
    for key in lst[1:]:
        print key
        temp=features_dc[key]
        final_merged_df=final_merged_df.join(temp, 'CallerId', 'outer')
        print 'after join count', final_merged_df.shape
    return final_merged_df


def merge_sorted_dfs_from_dict(features_dc):
    lst=[k  for k in features_dc.keys() if 'Date' not in k]
    final_merged_df=features_dc[lst[0]]
    print final_merged_df.shape
    for key in lst[1:]:
        for c in features_dc[key].column_names():
            if c!='CallerId':
                final_merged_df[c]=features_dc[key][c]
        print 'after join count', final_merged_df.shape
    return final_merged_df


def network_rank(raw_df,features_df):
    raw_df=raw_df[["ReceiverId","CallerId"]]
    joined_df=features_df.join(raw_df, 'CallerId', 'left')
    joined_df=joined_df.remove_column('ReceiverId')
    groupBy=['CallerId']
    cols=set(joined_df.column_names())-set(['CallerId','ReceiverId'])
    funcs=[agg.CONCAT]
    exprs={}
    for c in cols:
        for f in funcs:
            exprs[operations_names_dict[f]+'('+c+')']=f(c)
    df_ret=joined_df.groupby(key_columns=groupBy, operations=exprs)
    df_ret=df_ret.join(features_df, 'CallerId', 'left')
    for each in df_ret.column_names():
        if 'CallerId'!=each and 'joincollect' not in each:
            print each
            df_ret['rank('+each+')']=df_ret.apply(lambda x:_rank(x['joincollect('+each+')'],x[each]))
    for each in df_ret.column_names():
        if each!='CallerId' and 'rank' not in each:
            df_ret=df_ret.remove_column(each)
    return df_ret






def feature_name_counting(df, comb):
    print 'feature_name_counting', comb,' ',df.shape
    from collections import defaultdict
    header=df.column_names()
    lst=['(Duration)','(ReceiverCell)','(CallerCell)','(ReceiverId)','(CallerCell;ReceiverCell)','(Date)','(Day)']
    dc=defaultdict(int)
    dc_names=defaultdict(list)
    to_remove=[]
    for each in header:
        for pattern in lst:
            if pattern in each:
                dc[pattern]+=1
                dc_names[pattern].append(each)
                to_remove.append(each)
    header2=list(set(header)-set(to_remove))
    print 'Inside Feature name counting func: unmatched', header2
    print dc


def analyze(voice_df2,seeds_df,filter_name='', join_with_seeds=True):
    comb_params=[each for each in generate_pivots(voice_df2.column_names())]
    keyCol='CallerId'
    features_dc, features_dc_names={},{}
    for i,comb in enumerate(comb_params):
        print '*** i, comb', i, comb
        postfix=':'.join(x for x in comb)
        print '*** Base Aggregation'
        df_ret=apply_aggregate(voice_df2, comb, operations.keys(),operations , postfix)
        print '*** Shape after base aggregate', df_ret.shape
        print '*** Special Operations'
        df_ret=apply_special_operations(special_operations,comb, df_ret, postfix)
        print '*** Remove extra columns'
        df_ret=remove_column(df_ret, 'joincollect')
        print '*** Shape after special operations', df_ret.shape
        print '*** Reducing to the base level'
        if len(comb)>1:
            df_joined=composite_reduce(df_ret, comb)
        else:
            df_joined=df_ret
        # Reduce the extra columns from the df_joined
        for c in comb:
            if c!=keyCol:
                if c in df_joined.column_names():
                    df_joined=df_joined.remove_column(c)
        features_dc[tuple(comb)]=df_joined
        features_dc_names[tuple(comb)]=df_joined.column_names()
        print '*** Shape after composite reduce', df_joined.shape
        feature_name_counting(features_dc[tuple(comb)], comb)
        #print 'Column_names', df_joined.column_names()
        #df_joined.export_csv('features_temp'+postfix+'.csv')
        # Only calculating network rank for the first level features
    network_rank_df=network_rank(voice_df2, features_dc['CallerId',])
    network_rank_df=remove_column(network_rank_df, 'joincollect')
    print '*** shape of the network rank df is ', network_rank_df.shape
    #merged_df=merge_dfs_from_dict(features_dc)
    for each in features_dc:
        print each
        features_dc[each]=features_dc[each].sort('CallerId')
    network_rank_df=network_rank_df.sort('CallerId')
    # This join needs to be changed
    features_dc['network_rank_CallerId']=network_rank_df
    merged_df=merge_sorted_dfs_from_dict(features_dc)#.join(network_rank_df, on='CallerId', how='left')
    joined_df=align_for_network_aggregate(merged_df, voice_df2)
    joined_df=joined_df.sort('CallerId')
    callerids=joined_df['CallerId'].unique()
    chunks = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]
    if len(callerids)>2000000:
        callerIdList=chunks(callerids, len(callerids)/4)
        net_df_lst=[]
        for i in range(len(callerIdList)):
            joined_df_temp=joined_df.filter_by(callerIdList[i],'CallerId')
            net_df_temp=network_aggregate(joined_df_temp, postfix='Network')
            net_df_lst.append(net_df_temp)
        net_df=net_df_lst[0]
        for i in range(1,len(net_df_lst)):
            net_df=net_df.append(net_df_lst[i])
    else:
        net_df=network_aggregate(joined_df, postfix='Network')
    print '*** Shape of the network aggregate df ',net_df.shape
    merged_net_df=merged_df.join(net_df, on='CallerId', how='left')
    #merged_net_df=merged_df
    if join_with_seeds:
        merged_net_df=merged_net_df.join(seeds_df, on='CallerId', how='right')
    if filter_name!='':
        rename_dc={}
        for each in set(merged_net_df.column_names())-set(['CallerId']):
            rename_dc[each]=each.replace(',',';')
        merged_net_df=merged_net_df.rename(rename_dc)
        rename_dc={}    
        for each in set(merged_net_df.column_names())-set(['CallerId']):
            rename_dc[each]='Partition:'+filter_name+';'+each
        merged_net_df=merged_net_df.rename(rename_dc)
    print '*** Final Shape', merged_net_df.shape
    return merged_net_df, features_dc_names




dc_header={'DateTime':0,'CallerId':1,'ReceiverId':2,'Type':3,'CallerCell':4,'ReceiverCell':5, 'Duration':6}
operations_names_dict={agg.COUNT:'count',agg.SUM:'sum',agg.MEAN:'mean',agg.STD:'stddev',agg.CONCAT:'joincollect',
    agg.MIN:'min',agg.MAX:'max',_avgdiff:'avgdiff',_calc_rog:'calc_rog',
_diversity:'diversity', _avg_distance:'avg_distance',_unique_count:'unq'}
operations={'Duration':[agg.MEAN,agg.STD,agg.SUM],'ReceiverId':[agg.COUNT,agg.CONCAT],'CallerCell':[ agg.CONCAT],
    'ReceiverCell':[agg.CONCAT],'Day':[agg.CONCAT],'Date':[agg.CONCAT],'GeoDistance':[agg.COUNT,agg.MEAN,agg.STD,agg.MIN,agg.MAX,agg.SUM]}
special_operations={#'Date':[_avgdiff],
 'Day':[_unique_count],'ReceiverId':[_unique_count],
'CallerCell':[ _unique_count],
    'ReceiverCell':[ _unique_count]}
partition_columns=['Alter','Direction','IsWeekend','Type']
def swap_directions(sf):
    sf2=sf.copy()
    sf2=sf2.rename({'ReceiverId':'Temp','ReceiverCell':'TempCell'})
    sf2=sf2.rename({'CallerId':'ReceiverId','CallerCell':'ReceiverCell'})
    sf2=sf2.rename({'Temp':'CallerId','TempCell':'CallerCell'})
    sf2['Direction']='In'
    return sf2


if __name__=='__main__':
    if len(sys.argv)!=5:
        print 'Wrong ags, pass CDR file, seed file, key_for_output and output_dir'
        sys.exit(-1)
    fname=sys.argv[1] # CDR file
    sample=sys.argv[2] # seed file
    key = sys.argv[3] # key
    out_dir=sys.argv[4]
    df=gl.SFrame.read_csv(fname, column_type_hints=str)
    if 'sms' in fname:
        df['Duration']=0.0
    df['Duration']=df['Duration'].astype(float)
    df['GeoDistance']=0.0
    if 'CallerLAC' in df.column_names():
        df['CallerCell']=df.apply(lambda x:x['CallerCell']+'_'+x['CallerLAC'])
        df=df.remove_column('CallerLAC')
    if 'ReceiverLAC' in df.column_names():
        df['ReceiverCell']=df.apply(lambda x:x['ReceiverCell']+'_'+x['ReceiverLAC'])
        df=df.remove_column('ReceiverLAC')
    #df['Duration']=df['Duration'].astype(float)
    processDate(df)
    sample=gl.SFrame.read_csv(sample, header=False,  column_type_hints=df['CallerId'].dtype()).rename({'X1':'CallerId'})
    ret, features_dc_names=analyze(df,sample, key)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    ret.export_csv(out_dir+'//features_'+key+'.csv')
    

