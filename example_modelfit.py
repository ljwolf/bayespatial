
# coding: utf-8

# In[1]:


import theano.tensor as tt
import pysal as ps
import matplotlib.pyplot as plt
import seaborn.apionly as sns
import numpy as np
import pandas as pd
import ops
import distributions as spdist
import scipy.sparse as spar
import scipy.sparse.linalg as spla
import pymc3 as mc
plt.ion()

# In[2]:


df = ps.pdio.read_files(ps.examples.get_path('south.shp'))
df = df.query('STATE_NAME in ("Texas", "Oklahoma")')


# In[3]:


df.columns


# In[4]:


df['TEX'] = (df.STATE_NAME == 'Texas').astype(int)


# In[5]:


yname = 'FH90'
xnames = ['GI89', 'HR90', 'POL90', 'UE90', 'FH80', 'TEX']
Y = df[yname].values
X = df[xnames].values
N,P = X.shape


# In[6]:


sns.pairplot(pd.DataFrame([Y.flatten(),*X.T], index = [yname] + xnames).T)


# In[7]:


W = ps.weights.Queen.from_dataframe(df)


# In[8]:


W.transform = 'r'


# In[9]:


known_beta =  np.asarray([[4], [-2], [4], [-5], [1], [9], [20]])


# In[10]:


Yknown = known_beta[0] + X.dot(known_beta[1:])


# In[11]:
filt = np.eye(W.n) - .45 * W.sparse.toarray()
err = np.random.normal(0,1,size=(W.n,1))

efilt = np.linalg.solve(filt, err)
Yknown_e = Yknown + efilt
Yknown_l = np.linalg.solve(filt, Yknown + err)
Yknown_norm = Yknown + err


# In[12]:

se_pysal = ps.spreg.ML_Error(Yknown_e, 
                             X, w=W, 
                             name_y=yname, name_x=xnames) 
sl_pysal = ps.spreg.ML_Error(Yknown_l, 
                             X, w=W, 
                             name_y=yname, name_x=xnames)

# In[13]:


print(se_pysal.summary)


# In[14]:


print(sl_pysal.summary)


# In[15]:


import ord as eigval_dists
import imp
imp.reload(eigval_dists)


# In[16]:


evals = np.linalg.eigvals(W.sparse.toarray())


# In[17]:


emin, emax = evals.min(), evals.max()


# In[18]:


# In[19]:


with mc.Model() as standard:
    intercept = mc.Normal('intercept', 0, sd=10, testval=0)
    slopes = mc.Normal('slopes', 0, sd=10, shape=P)
    mean = intercept + tt.dot(X, slopes)
    scale = mc.HalfCauchy('scale', 5)
    
    outcome = mc.Normal('outcome', mu=mean, sd=scale, observed=Yknown_norm.flatten())


# In[20]:


with standard:
    samp = mc.sample(500, progressbar=False)


# In[21]:


mc.traceplot(samp)


# In[22]:


samp['scale'].mean()


# In[26]:


with mc.Model() as SE:
    intercept = mc.Normal('intercept', 0,sd=10, testval=0)
    slopes = mc.Normal('slopes', 0,sd=10, shape=P)
    s2 = mc.HalfCauchy('scale', 5)
    autoreg = mc.Bound(mc.Normal, lower=1/emin, upper=1/emax)                               ('autoreg', mu=0, sd=1, testval=0)
    mean = intercept + tt.dot(X, slopes)
    outcome = eigval_dists.SAR_Error('outcome', mean=mean, scale=s2, 
                                     autoreg=autoreg, weights=W, eigs=evals,
                                     observed=Yknown_e)


# In[27]:


with SE:
    samp = mc.sample(500, progressbar=False)


# In[28]:


mc.traceplot(samp)


# In[ ]:




