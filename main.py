from decisionTree import tree, classify, Candidate

print("Hello, you are using interview helper for X's company, fulfill data to see if candidate is good enough to work here")
level = input("Level of candidate is [Junior, Mid, Senior]:   ")
lang = input("Candidate's main programming language is [C#, JavaScript, Python, Ruby, Java, C++]: ")
social_media = input("Is Candidate active in SocialMedia? [True, False]: ")
education = input("Does candidate have higher education[True, False]: ")

answer = classify(tree, Candidate(level, lang, bool(social_media), bool(education)))

if answer == True:
    print("Candidate is good for our company")
else:
    print("Candidate isn't good enough")


