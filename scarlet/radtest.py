import numpy as np
is_rad=np.array([1,1,0,0,1,0,1])
N=len(is_rad)
rad_zones=[]
rad_zone=[]
for idx in range(N):
    if is_rad[idx]:
        rad_zone.append(idx)
    else:
        if len(rad_zone)>0:
            rad_zones.append(np.array(rad_zone))
            rad_zone=[]
if len(rad_zone)>0:
    rad_zones.append(np.array(rad_zone))
    rad_zone=[]
print(is_rad)
print(rad_zones)
