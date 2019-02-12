import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from matplotlib.colors import ListedColormap

####################################

train = np.array(      [[+1,+2],
                        [+2,+2],
                        [-1,-2],
                        [-2,-2],
                        [-2,-1] ] )

class_train = [0,0,1,1,1]
                        
test = np.array(       [[+1,+2],
                        [+2,+2],
                        [-1,-2],
                        [-2,-2],
                        [-2,-1] ] )

class_test  = [0,0,1,1,1]

####################################

svc = svm.SVC(kernel='linear')

svc.fit(train, class_train)
                 
predicted = svc.predict(test)
score = svc.score(test, class_test)

print('============================================')
print('\nScore ', score)
print('\nResult Overview\n',   metrics.classification_report(class_test, predicted))
print('\nConfusion matrix:\n', metrics.confusion_matrix(class_test, predicted)      )          
                
##########################################                
cmap, cmapMax = plt.cm.RdYlBu, ListedColormap(['#FF0000', '#0000FF'])           
                
fig = plt.figure()
ax = fig.add_subplot(1,1,1)     
       
       
h = 0.3  
x_min, x_max = train[:, 0].min()-.3, train[:, 0].max()+.3
y_min, y_max = train[:, 1].min()-.3, train[:, 1].max()+.3                
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))    
   
if hasattr(svc, "decision_function"):
        Z = svc.decision_function(np.c_[xx.ravel(), yy.ravel()])
else:
        Z = svc.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.7)

# Plot also the training points
ax.scatter(train[:, 0], train[:, 1], c=class_train, cmap=cmapMax)
# and testing points
ax.scatter(test[:, 0], test[:, 1], c=class_test, cmap=cmapMax, alpha=0.5)

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
plt.title(str(score))

plt.show()