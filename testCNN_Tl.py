from numpy import load
from keras.models import load_model
model = load_model('modelCNN_1.h5')

test=load('testing_data_1.npy')
sol=model.predict(test)
save('prediction_1',sol)
print(sol)  
