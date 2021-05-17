from numpy import load
from keras.models import load_model
model = load_model('modelANN_final.h5')

test=load('testing_data_finalANN.npy')
sol=model.predict(test)
save('prediction_finalANN',sol)
print(sol)  
