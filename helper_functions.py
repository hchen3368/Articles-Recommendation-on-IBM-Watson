import pandas as pd
import numpy as np



# create the user-article matrix with 1's and 0's

def create_user_item_matrix(df):
    '''
    INPUT:
    df - pandas dataframe with article_id, title, user_id columns
    
    OUTPUT:
    user_item - user item matrix 
    
    Description:
    Return a matrix with user ids as rows and article ids on the columns with 1 values where a user interacted with 
    an article and a 0 otherwise
    '''
    user_item_df = df.groupby(['user_id','article_id']).count().astype('int').unstack()
    user_item_df = (~user_item_df.isna())*1
    user_item_df = user_item_df.droplevel(0,axis=1)
    
    
    user_item = user_item_df.values
    
    
    return user_item, user_item_df


# create the user-article matrix with count values (trimmed)

def create_user_item_matrix_2(df, cap=4):
    '''
    INPUT:
    df - pandas dataframe with article_id, title, user_id columns
    cap - integer, the cap value in the matrix.
    log_transform - boolean, if true then apply log1p transform.
    
    OUTPUT:
    user_item - user item matrix 
    
    Description:
    Return a matrix with user ids as rows and article ids on the columns with 1 values where a user interacted with 
    an article and a 0 otherwise
    '''
    # user item matrix with count values
    user_item_df = df.groupby(['user_id','article_id']).count().astype('int').unstack()
    # index column by item (row by user)
    user_item_df = user_item_df.droplevel(0, axis=1)
    # trim values larger than cap
    user_item_df = user_item_df.applymap(lambda x: min(x,cap))
        
    
    return user_item_df