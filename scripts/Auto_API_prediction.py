#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud prediction  
# APIfied with GCP
# 
# Dataset available at https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download
# 
# Discussion with Gemini AI at https://gemini.google.com/app/da520b9418e5c083

# ## Model training

# In[1]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score

import pickle
import os


# In[2]:


data = pd.read_csv(os.path.join("data", "creditcard.csv"))


# In[3]:


data.head()


# In[4]:


X = data.drop('Class', axis=1)
y = data['Class']
X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.2, random_state=42)


# In[5]:


halfway = int(np.floor(len(X_holdout.index)/2))
X_test=X_holdout[:halfway]
y_test=y_holdout[:halfway]

X_validation=X_holdout[halfway:]
y_validation=y_holdout[halfway:]


# In[6]:


X_train


# In[7]:


scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[8]:


model = KNeighborsClassifier()
model.fit(X_train, y_train)


# In[9]:


score=roc_auc_score(y_test, model.predict(X_test))
score


# ## Model Upload

# In[10]:


with open(os.path.join("models", 'model.pkl'), 'wb') as f:
  pickle.dump(model, f)


# In[14]:


get_ipython().system('gsutil -m cp -r models gs://alfsn-models/models')


# ## Model deployment

# In[16]:


import google.cloud.aiplatform as aiplatform


# In[37]:


get_ipython().system('gcloud auth application-default login')


# In[51]:


def create_and_deploy_endpoint(project_id, region, model_dir, endpoint_name, machine_type, min_replica_count, max_replica_count):
  """  Creates a Vertex AI Endpoint and deploys a model to it.
  This function was created with the help of Gemini AI
  
  Args:
    project_id: The Google Cloud project ID.
    region: The region where the endpoint will be created.
    model_name: The name of the Vertex AI model.
    endpoint_name: The desired name for the endpoint.
    machine_type: The machine type for the endpoint instances.
    min_replica_count: The minimum number of replicas.
    max_replica_count: The maximum number of replicas.
  """
  # init AI client
  aiplatform.init(project=project_id, location=region)

  # create model from bucket
  model = aiplatform.Model.upload(
      display_name=f"imported_{endpoint_name}",
      artifact_uri=model_dir,
      serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-23:latest"
  )
  print(f"Model imported: {model.resource_name}")
  
  # creates endpoint
  endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)

  # creates a deployment spec
  deployed_model = endpoint.deploy(
      model=model, # this is a google.cloud.aiplatform.Model
      machine_type=machine_type,
      min_replica_count=min_replica_count,
      max_replica_count=max_replica_count,
      traffic_percentage=100,
      sync=True
  )

  return endpoint


# In[56]:


project_id = "lang-prompt"
region = "us-central1"
model_dir = "gs://alfsn-models/models"
endpoint_name = "KNN_fraud_detect"
machine_type = "n1-standard-2"
min_replica_count = 1
max_replica_count = 2


# In[57]:


endpoint = create_and_deploy_endpoint(project_id, 
                                      region, 
                                      model_dir, 
                                      endpoint_name, 
                                      machine_type,
                                      min_replica_count,
                                      max_replica_count)


# In[58]:


print(f"Endpoint created: {endpoint.resource_name}")


# ## Model access

# In[93]:


def predict_old(project_id: str, location: str, endpoint_id: str, instance_data: list):
    # Initialize Vertex AI client
    aiplatform.init(project=project_id, location=location)

    # Load the endpoint
    endpoint = aiplatform.Endpoint(endpoint_id)

    # Make a prediction
    prediction = endpoint.predict(instances=instance_data)

    return prediction


# In[94]:


def predict(project_id: str, 
            location: str, 
            endpoint_id: str, 
            instance_data: pd.DataFrame):
    """
    The function hits a Vertex API with a pandas DataFrame
    """
    aiplatform.init(project=project_id, location=location)

    endpoint = aiplatform.Endpoint(endpoint_id)

    # Convert DataFrame to list of dictionaries
    instances = instance_data.to_dict(orient='records')

    prediction = endpoint.predict(instances=instances)

    return prediction


# In[103]:


prediction = predict_old(project_id= project_id,
                    location=region, 
                    endpoint_id=endpoint.resource_name, 
                    instance_data=X_validation.head(10).to_dict('records') # send only some requests 
                    )


# In[91]:


prediction = predict(project_id= project_id,
                    location=region, 
                    endpoint_id=endpoint.resource_name, 
                    instance_data=X_validation.head(10) # send only some requests 
                    )
prediction


# In[ ]:




