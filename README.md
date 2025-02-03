# day_to_night_comparation

## Image Processing Approach

<div align="center">
    <a href="./">
        <img src="./figure/rev_framework.png" width="79%"/>
    </a>
</div>

<div align="center">
    <a href="./">
        <img src="./figure/syn_image2.png" width="79%"/>
    </a>
</div>

## Deep Learning Approach

<div align="center">
    <a href="./">
        <img src="./figure/edgefeatureGAN.png" width="65%"/>
    </a>
</div>

<div align="center">
    <a href="./">
        <img src="./figure/FIG411.png" width="65%"/>
    </a>
</div>

<div align="center">
<table><thead>
  <tr>
    <th rowspan="2"></th>
    <th colspan="2">FID</th>
    <th colspan="2">SSIM</th>
  </tr>
  <tr>
    <th>Train</th>
    <th>Test</th>
    <th>Train</th>
    <th>Test</th>
  </tr></thead>
<tbody>
  <tr>
    <td>Vanilla CycleGAN</td>
    <td>35,61</td>
    <td>62,54</td>
    <td>0,44</td>
    <td>0,38</td>
  </tr>
  <tr>
    <td>FPN-CycleGAN</td>
    <td>71,64</td>
    <td>104,46</td>
    <td>0,48</td>
    <td>0,44</td>
  </tr>
  <tr>
    <td>UVCGAN</td>
    <td>13,67</td>
    <td>16,68</td>
    <td>0,49</td>
    <td>0,42</td>
  </tr>
  <tr>
    <td>UVCGAN with Edge Feature Loss</td>
    <td>21,83</td>
    <td>47,79</td>
    <td>0,45</td>
    <td>0,40</td>
  </tr>
</tbody>
</table>
</div>
