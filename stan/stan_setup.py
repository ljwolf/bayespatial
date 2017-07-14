import pystan as st
import pysal as ps
import numpy as np

df = ps.pdio.read_files(ps.examples.get_path('south.shp'))
df = df.query('STATE_NAME in ("Texas", "Oklahoma", "Arkansas", "Louisiana")')
df = df[df.HR90 > 0]
Y = np.log(df[['HR90']].values)
X = df[['GI89', 'FH90']].values
N,P = X.shape

W = ps.weights.Queen.from_dataframe(df)
W.transform = 'r'
Wm = W.sparse.toarray()
Yknown = 4 + X.dot(np.asarray([[-2], [4]]))
Yknown_f = Yknown + np.linalg.solve((np.eye(W.n) - .45 * W.sparse.toarray()), 
                                    np.random.normal(0,1,size=(W.n,1)))

evalues = np.linalg.eigvals(Wm)

data = dict(X = X, y = Yknown_f.flatten(), n = N, p=P, W=Wm, evalues=evalues)


#test = st.stan(file='./se.stan', data = data, iter=1000) 
