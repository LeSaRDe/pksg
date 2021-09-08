
JFLAGS = -g -d bin -sourcepath src
JC = javac
RM = rm

SRCPATH = src/core_nlp_sentiments

.SUFFIXES: .java .class

.java.class:
	$(JC) $(JFLAGS) $*.java

CLASSES = \
        $(SRCPATH)/GlobalSettings.java \
		$(SRCPATH)/CoreNLPSentiments.java \
		$(SRCPATH)/PhraseSentimentTask.java \
		$(SRCPATH)/PhraseSentimentParallel.java

default: classes

classes: $(CLASSES:.java=.class)

env:
	export CLASSPATH=./bin:${CLASSPATH}

clean:
	$(RM) -rf bin/*
