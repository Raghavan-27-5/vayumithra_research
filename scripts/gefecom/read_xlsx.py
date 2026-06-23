import zipfile, xml.etree.ElementTree as ET
# xlsx is just a zip of XML files
wb_path = r'C:\Projects\raghavan\GEFCom2014\GEFCom2014 Data\Provisional_Leaderboard_V2.xlsx'
z = zipfile.ZipFile(wb_path)
print('Files in xlsx:')
for n in z.namelist():
    print('  ' + n)
print()
# Read shared strings
if 'xl/sharedStrings.xml' in z.namelist():
    tree = ET.parse(z.open('xl/sharedStrings.xml'))
    root = tree.getroot()
    ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    strings = [si.find('.//s:t', ns).text if si.find('.//s:t', ns) is not None else '' for si in root.findall('.//s:si', ns)]
    print('First 50 shared strings:')
    for i, s in enumerate(strings[:50]):
        print(f'  [{i}] {s}')
