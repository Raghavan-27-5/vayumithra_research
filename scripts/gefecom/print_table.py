import numpy as np
ours = [0.10875, 0.20410, 0.13666, 0.12526, 0.14158, 0.21555, 0.10614, 0.15443, 0.14425, 0.17111, 0.20814, 0.15076]
bench = [0.07536, 0.07389, 0.08195, 0.07897, 0.08425, 0.07694, 0.09949, 0.11871, 0.09958, 0.09583, 0.07614, 0.07936]
kpower = [0.03562, np.nan, 0.03888, 0.03069, 0.03889, 0.03190, 0.03635, 0.03710, 0.03879, 0.04168, 0.03917, 0.03915]
months = ['Oct 2012','Nov 2012','Dec 2012','Jan 2013','Feb 2013','Mar 2013','Apr 2013','May 2013','Jun 2013','Jul 2013','Aug 2013','Sep 2013']

print(f"Week  Month       Ours     Bench    kPower       O/B")
print('-' * 53)
for i in range(12):
    r = ours[i] / bench[i]
    kp = f'{kpower[i]:.5f}' if not np.isnan(kpower[i]) else '  N/A  '
    print(f'{i+1:>4}  {months[i]:>10}  {ours[i]:>8.5f}  {bench[i]:>8.5f}  {kp:>8}  {r:>7.2f}x')
avg_o = np.mean(ours)
avg_b = np.mean(bench)
avg_k = np.nanmean(kpower)
print('-' * 53)
print(f'Avg  {"":>10}  {avg_o:>8.5f}  {avg_b:>8.5f}  {avg_k:>8.5f}  {avg_o/avg_b:>7.2f}x')
