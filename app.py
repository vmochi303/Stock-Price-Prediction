import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense,Dropout,LSTM
from keras.models import Sequential
# from keras.models import lode_model
from tensorflow import keras


import streamlit as st

start ='2010-01-01'
end='2019-12-31'

st.title('Stock Trend Prediction')


user_input=st.text_input('Enter Stock Ticker','AAPL')
start=st.text_input('start','2010-01-01')
end=st.text_input('end','2019-12-31')
df=data.DataReader(user_input,'yahoo',start,end)


# describing data 
st.subheader('Data from '+start+' - '+end) #1
st.write(df.describe())

st.subheader('Closing price vs Time Chart') #2
fig  = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)


#100 days
st.subheader('Closing price vs Time Chart with 100DA') #3
ma100=df.Close.rolling(100).mean()
fig  = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)


#200 days
st.subheader('Closing price vs Time Chart with 100DA & 200DA') #3
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig2  = plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close)
st.pyplot(fig2)


# #300 days
st.subheader('Closing price vs Time Chart with 100DA .. 300DA') #3
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
ma300=df.Close.rolling(300).mean()
fig3  = plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(ma300,'p')
plt.plot(df.Close)
st.pyplot(fig3)

# #400 days
st.subheader('Closing price vs Time Chart with 100DA .. 400DA') #3
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
ma300=df.Close.rolling(300).mean()
ma400=df.Close.rolling(400).mean()
fig4  = plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(ma300,'p')
plt.plot(ma400,'m')
plt.plot(df.Close)
st.pyplot(fig4)


# #500 days
st.subheader('Closing price vs Time Chart with 100DA .. 500DA') #3
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
ma300=df.Close.rolling(300).mean()
ma400=df.Close.rolling(400).mean()
ma500=df.Close.rolling(500).mean()
fig5  = plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(ma300,'p')
plt.plot(ma400,'m')
plt.plot(ma500,'y')
plt.plot(df.Close)
st.pyplot(fig5)

data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))


data_training_array=scaler.fit_transform(data_training)

 
x_train=[]
y_train=[]

for i in range(100,data_training_array.shape[0]):
    x_train.append(data_training_array[i-100 :i])
    y_train.append(data_training_array[i,0])


x_train,y_train=np.array(x_train),np.array(y_train)

model = keras.models.load_model('keras_model.h5')
try:
    past_100_days=data_training.tail(100)
except :
    pass
final_df=past_100_days.append(data_testing,ignore_index=True)
input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]
for i in range(100,input_data.shape[0]):
  x_test.append(input_data[i-100 :i])
  y_test.append(input_data[i,0])



st.subheader('Prediction vs Original')

x_test,y_test=np.array(x_test),np.array(y_test)

y_predicted=model.predict(x_test)


diff = []
z=0 # count days
for i,j in zip(y_test,y_predicted):
    diff.append(i-j)
    z+=1
s = 0 #sum
for i in diff:
    s+=i 
result  =(s/z) #result cantains difference 


fig = plt.figure(figsize=(12,6))
scaler = scaler.scale_
scale_fector=1/scaler[0]
y_predicted=y_predicted * scale_fector
y_test=y_test*scale_fector

fig6 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label="Original Price")
plt.plot(y_predicted,'r',label="Predicted Price")
# plt.plot(diff,'y',label="Difference")

plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig6)
print(f"Differnece is :- {result}")