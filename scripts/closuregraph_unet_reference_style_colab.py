# Google Colab: run this cell/file, then download the PNG and PDF from /content.
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch, FancyArrowPatch, Circle

OUT='/content/ClosureGraph_InSAR_Figure1_reference_style'
os.makedirs(os.path.dirname(OUT), exist_ok=True)
navy='#183b65'; blue='#4f81bd'; orange='#f29f3f'; teal='#2b9c8f'; purple='#8b6bb8'

fig=plt.figure(figsize=(18,15),dpi=800); fig.patch.set_facecolor('white')
def panel(rect,label):
 ax=fig.add_axes(rect); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
 ax.add_patch(FancyBboxPatch((.01,.02),.98,.95,boxstyle='round,pad=.01,rounding_size=.04',fill=False,ec=navy,lw=1.8,linestyle=(0,(3,2))))
 ax.text(.035,.91,label,fontsize=16,weight='bold',color=navy); return ax
def arrow(ax,x1,y1,x2,y2,c=navy): ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=13,lw=1.2,color=c))
def route(ax, points, c=purple):
 """Orthogonal skip path with one clean arrowhead at its destination."""
 for (x1,y1),(x2,y2) in zip(points[:-2],points[1:-1]):
  ax.plot([x1,x2],[y1,y2],color=c,lw=1.05)
 x1,y1=points[-2]; x2,y2=points[-1]
 arrow(ax,x1,y1,x2,y2,c)
def volume(ax,x,y,w,h,d,color,label=None):
 # perspective stack of feature maps
 for i in range(5):
  xx=x+i*d*.26; yy=y+i*d*.26
  ax.add_patch(Polygon([(xx,yy),(xx+w,yy),(xx+w+d,yy+d),(xx+d,yy+d)],fc=color,ec=navy,lw=.45,alpha=.68))
  ax.add_patch(Polygon([(xx+w,yy),(xx+w,yy+h),(xx+w+d,yy+h+d),(xx+w+d,yy+d)],fc=color,ec=navy,lw=.45,alpha=.42))
  ax.add_patch(Polygon([(xx,yy),(xx+w,yy),(xx+w,yy+h),(xx,yy+h)],fc=color,ec=navy,lw=.5,alpha=.72))
 if label: ax.text(x+w/2,y-.06,label,ha='center',fontsize=10,weight='bold')

# a: compact, reference-style U-shaped five-level encoder-decoder
ax=panel([.05,.56,.90,.37],'(a)  ClosureGraph-InSAR reconstruction architecture')
ax.text(.50,.84,'Five-level separable 3D U-Net with aligned skip connections',ha='center',fontsize=14,weight='bold',color=navy)
volume(ax,.05,.36,.055,.30,.021,'#b9d7ef','Observed LOS\nX, M')
enc=[(.15,.33,.052,.36,'16'),(.245,.38,.048,.26,'32'),(.335,.43,.044,.16,'64'),(.415,.47,.040,.10,'128'),(.490,.495,.034,.055,'256')]
dec=[(.570,.47,.040,.10,'128'),(.650,.43,.044,.16,'64'),(.740,.38,.048,.26,'32'),(.830,.33,.052,.36,'16')]
for x,y,w,h,l in enc: volume(ax,x,y,w,h,.016,orange,l)
for x,y,w,h,l in dec: volume(ax,x,y,w,h,.016,'#76b8b5',l)
for a,b in zip(enc[:-1],enc[1:]): arrow(ax,a[0]+a[2]+.018,a[1]+a[3]/2,b[0]-.006,b[1]+b[3]/2)
for a,b in zip(dec[:-1],dec[1:]): arrow(ax,a[0]+a[2]+.018,a[1]+a[3]/2,b[0]-.006,b[1]+b[3]/2)
arrow(ax,.112,.51,.145,.51); arrow(ax,.530,.522,.565,.522)
# Four parallel routes: each begins above the encoder map and lands directly above its matched decoder map.
for (e,d,y) in zip(enc[:4], dec[::-1], [.785,.755,.725,.695]):
 sx=e[0]+e[2]/2; sy=e[1]+e[3]+.012
 dx=d[0]+d[2]/2; dy=d[1]+d[3]+.012
 route(ax,[(sx,sy),(sx,y),(dx,y),(dx,dy)],purple)
ax.text(.50,.765,'skip connections',ha='center',fontsize=9,color=purple,style='italic')
ax.add_patch(FancyBboxPatch((.908,.44),.062,.115,boxstyle='round,pad=.02',fc='#e7f7f4',ec=teal,lw=1.4))
ax.text(.939,.497,'Reliability-\nweighted graph\n{1,2,4,8}',ha='center',va='center',fontsize=6.7,weight='bold')
arrow(ax,.895,.51,.906,.51,teal)
ax.add_patch(FancyBboxPatch((.910,.225),.058,.062,boxstyle='round,pad=.018',fc='#fff2df',ec='#d87b00',lw=1.3))
ax.text(.939,.256,'Reconstructed\nLOS sequence',ha='center',va='center',fontsize=5.9,weight='bold')
arrow(ax,.939,.44,.939,.288,teal)
ax.text(.50,.08,'Spatial depthwise 1×3×3  →  temporal pointwise 3×1×1  •  BatchNorm + PReLU  •  matched baseline omits graph refinement',ha='center',fontsize=9.5)

# b original convolution
ax=panel([.05,.31,.90,.20],'(b)  Original 3D convolution operation')
ax.text(.50,.80,'Original 3D convolution',ha='center',fontsize=11,weight='bold')
volume(ax,.10,.29,.13,.34,.030,'#256eb5','Input tensor\nC × D × W × L')
volume(ax,.45,.34,.065,.16,.020,'#d4a6db','Filter\nK × K × K')
volume(ax,.77,.29,.13,.34,.030,'#256eb5','Output tensor\nF × D × W × L')
arrow(ax,.28,.46,.43,.46); arrow(ax,.55,.46,.75,.46)
ax.text(.50,.14,'A full 3D kernel jointly learns spatial and temporal context.',ha='center',fontsize=9.5,color='#43546b')

# c depthwise / pointwise
ax=panel([.05,.06,.90,.20],'(c)  Separable 3D convolution used in ClosureGraph-InSAR')
ax.text(.50,.80,'Spatial depthwise convolution followed by temporal pointwise convolution',ha='center',fontsize=10.5,weight='bold')
volume(ax,.07,.29,.13,.34,.030,'#256eb5','Input tensor')
volume(ax,.35,.29,.13,.34,.030,orange,'Depthwise\n1 × 3 × 3')
volume(ax,.66,.29,.13,.34,.030,'#5bb7ad','Pointwise\n3 × 1 × 1')
arrow(ax,.24,.46,.34,.46); arrow(ax,.52,.46,.65,.46)
ax.add_patch(FancyBboxPatch((.83,.39),.10,.13,boxstyle='round,pad=.02',fc='#edf3fb',ec=blue,lw=1.1))
ax.text(.88,.455,'BatchNorm\n+ PReLU',ha='center',va='center',fontsize=8.2,weight='bold',color=navy)
arrow(ax,.81,.46,.83,.46)
ax.text(.50,.14,'This factorisation separates spatial filtering from temporal mixing.',ha='center',fontsize=9.5,color='#43546b')
fig.savefig(OUT+'.png',dpi=800,bbox_inches='tight'); fig.savefig(OUT+'.pdf',bbox_inches='tight')
print('Saved:',OUT+'.png and .pdf')
