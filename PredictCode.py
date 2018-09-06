from sklearn.externals import joblib
model = joblib.load('model.pkl') 
def Predict_Code(s):
    pr_label = model.predict(vectorizer.transform([s]))
    print(pr_label)
    print("######################################")
    if pr_label == "searchMessage":
        f = open('searchMessage.txt', 'r')
        print (f.read())
        f.close()
    elif pr_label == "PressEnter":
        f = open('PressEnter.txt', 'r')
        print (f.read())
        f.close()
    elif pr_label == "Link_Element_click":
        f = open('Link_Element_click.txt', 'r')
        print (f.read())
        f.close()
    elif pr_label == "PressF4":
        f = open('PressF4.txt', 'r')
        print (f.read())
        f.close()
    elif pr_label == "PressSave":
        f = open('PressSave.txt', 'r')
        print (f.read())
        f.close()
    elif pr_label == "PressUIBBDeleteButton":
        f = open('PressUIBBDeleteButton.txt', 'r')
        print (f.read())
        f.close()
    elif pr_label == "PressInitialize":
        f = open('PressInitialize.txt', 'r')
        print (f.read())
        f.close()
    elif pr_label == "CheckValue_SAP":
        f = open('CheckValue_SAP.txt', 'r')
        print (f.read())
        f.close()
    elif pr_label == "PressCopyPolicy":
        f = open('PressCopyPolicy.txt', 'r')
        print (f.read())
        f.close()
    elif pr_label == "PressComplete":
        f = open('PressComplete.txt', 'r')
        print (f.read())
        f.close()
    elif pr_label == "PressCancel":
        f = open('PressCancel.txt', 'r')
        print (f.read())
        f.close()
    elif pr_label == "PressAdd":
        f = open('PressAdd.txt', 'r')
        print (f.read())
        f.close()
    elif pr_label == "Mark":
        f = open('Mark.txt', 'r')
        print (f.read())
        f.close()
    elif pr_label == "SetPSBasicData":
        f = open('SetPSBasicData.txt', 'r')
        print (f.read())
        f.close()
    elif pr_label == "NewPolicySection":
        f = open('NewPolicySection.txt', 'r')
        print (f.read())
        f.close()
    elif pr_label == "New_Policy":
        f = open('New_Policy.txt', 'r')
        print (f.read())
        f.close()
    elif pr_label == "PolicySearch":
        f = open('PolicySearch.txt', 'r')
        print (f.read())
        f.close()