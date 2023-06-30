import streamlit as st
import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import QED, Descriptors, Lipinski

##### 페이지레이아웃 지정 단락 #####
st.set_page_config(
    page_title="PT2 WEB-app",
    layout="wide"
)

##### 함수 지정 단락 #####
def calc_rdkit(query):
    """
    간단한 SMILES 계산기.
    rdkit으로 물리화학적 특성을 반환시킴
    mw    : ExactMolWt과 약간 다르다.
    qed   : https://doi.org/10.1038/nchem.1243
    wlogp : S.A. Wildman and G. M. Crippen JCICS 39 868-873 (1999)
    tpsa  : https://doi.org/10.1021/jm000942e
            Swiss ADME에서 계산하는 TPSA는 Etrl 2000. 에 기반한다. RDKit은 이 참고문헌에 S와 P의 영향을 함께 계산할 수 있는 옵션을 제공한다. (includeSandP=True) 
            includeSandP 옵션을 켜지 않으면 SwissADME에서 계산한 결과와 동일하다.
    Ro5   : https://doi.org/10.1016/S0169-409X(96)00423-1
    lipinski module : https://www.rdkit.org/docs/source/rdkit.Chem.Lipinski.html?highlight=lipinski%20module
    """

    mol = Chem.MolFromSmiles(query)
    mw = Descriptors.MolWt(mol)
    qed = round(QED.qed(mol),3)
    wlogp = Descriptors.MolLogP(mol) #Chem.Crippen.MolLogP(mol)과 같은 기능
    tpsa = Descriptors.TPSA(mol, includeSandP = False)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rtb = Lipinski.NumRotatableBonds(mol)
    violation = ro5(mw, hbd, hba, wlogp)

    return mw, qed, wlogp, tpsa, hbd, hba, rtb, violation


def ro5(mw, hbd, hba, wlogp):
    ro5_violoations = 0
    failed = list()
    if mw >= 500:
        ro5_violations = ro5_violations + 1
        failed.append("molecular weight violated: %s" % mw)
    if hbd > 5:
        ro5_violations = ro5_violations + 1
        failed.append("HBD violated: %s" % hbd)
    if hba > 10:
        ro5_violations = ro5_violations + 1
        failed.append("HBA violated: %s" % hba)
    if wlogp > 5:
        ro5_violations = ro5_violations + 1 #초과냐 이상이냐

    if ro5_violations > 0:
        violation = "violated"
    else:
        violation = "passed"
    
    return result

##### 사이드바 지정 단락 #####
with st.sidebar:
    st.write("여기는 사이드바")


##### 메인콘텐츠 지정 단락 #####
input_string  = st.text_input("Please input interesting SMILES","CC(=C)C(O)=O", help="올바르지 않은 SMILES일 경우 에러가 출력됩니다.")
st.write("입력한 분자 SMILES: ", input_string)

mw, qed, wlogp, tpsa, hbd, hba, rtb, violation= calc_rdkit(input_string)
st.write("Molecular weight: ", mw)
st.write("QED: ", qed)
st.write("wlogp: ", wlogp)
st.write("TPSA; ", tpsa, "(단, S와 P의 영향은 고려하지 않았으며 SwissADME의 결과와 동일함.)")
st.write("num of Hydrogen bond donors; ", hbd)
st.write("num of Hydrogen bond acceptors; ", hba)
st.write("num of Rotatable bonds; ", rtb)
st.write("Ro5; ", violation)
