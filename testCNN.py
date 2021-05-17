from numpy import load
from keras.models import load_model
model = load_model('modelCNN_3.h5')

test=load('testing_data_3.npy')
sol=model.predict(test)
save('prediction_3',sol)
print(sol)  
