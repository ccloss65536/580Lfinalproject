#Carl Closs Brian Grant Kevin Yan 
SHELL := /bin/bash
NUM = FINAL
HEADERS = 
COMPILE = g++
FLAGS = -g  -Wall -Wextra -Wno-unused-parameter -lrt -pthread -lm
NAME1 = training
NAME2 = 
FILE = ccloss1_$(NUM).tar.gz 
TESTOPTS = lol
DEBUG_OPTS = --silent -x cmds.txt
all: $(NAME1)
debug: $(NAME1)
	gdb $(NAME1) $(DEBUG_OPTS)
push:
	#@read -p "commit message (input ctrl+C to stop the push process, 1 line only): " MESSAGE
	git add -A
	git commit #-m "$(MESSAGE)"
	git push 
	@#Only in bash, read can have a prompt,
	@#and put the entire imput string into an enviroment variable called $REPLY
$(NAME1): $(NAME1).cpp 
	$(COMPILE) -c $(FLAGS)  $(NAME1).cpp 
	$(COMPILE) $(FLAGS) $(NAME1).o -o  $(NAME1)
$(NAME2): $(NAME2).cpp
	$(COMPILE) -c $(FLAGS) $(NAME2).cpp
	$(COMPILE) $(FLAGS) $(NAME2).o -o $(NAME2)
clean: backup
	rm -f *.o *.swp *.gch .go* $(NAME1) .nfs*
backup:
	cd .. && 	tar -cvzf  $(shell date +%s)_$(FILE) 580Lfinalproject
	mv ../$(shell date +%s)_$(FILE) ~/backups
submit: $(NAME1) clean
	cd .. && 	tar -cvzf  $(FILE) 580Lfinalproject
ifneq "$(findstring remote, $(HOSTNAME))"  "remote"
		firefox submit.htm
else 
		mutt -s "Prog$(NUM)_submission" ccloss1@binghamton.edu <submit.htm -a ../$(FILE)
endif
	#hack to determine whether we should use firefox or email to self
