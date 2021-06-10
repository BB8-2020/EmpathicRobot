# Models 
In this folder you can find all the models that we have trained. The analysis of each model can also be found here.

## Content
- [Baseline model](https://github.com/BB8-2020/EmpathicRobot/tree/main/models/baseline_model) 
    
    Simple sequential model that can recognize whether someone is happy or not. 
    
- [Classification model](https://github.com/BB8-2020/EmpathicRobot/tree/main/models/classification_model) 
     
    **Conv**: 
  
    A sequential model that tries to recognize the following 7 emotions.
  
          1. Neutral
          2. Happy
          3. Surprise
          4. Sadness
          5. Anger
          6. Disgust
          7. Fear
    
   **VGG16**: 
  
    A pre-trained model often used to recognize emotions. For more information about this model check 
    this [keras](https://keras.io/api/applications/) page.
    We use here both the pre-trained version and the model without weights. 
    Which makes us train the model ourselves with our own photos.
    We are still trying to recognize the same 7 emotions.
  

- [Validation model](https://github.com/BB8-2020/EmpathicRobot/tree/main/models/validation_model)
  
    **Dit moet nog bijgewerkt worden!!**
  
## Installation 

To run our models you need to get the data first. Therefore you can have a look at [this](https://github.com/BB8-2020/EmpathicRobot/tree/main/data) page.

Sinds you have the data, place it into a folder called `data` in `models`. 
After that you will be abel to run the code of every model.