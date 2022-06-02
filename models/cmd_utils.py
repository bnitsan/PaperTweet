import getopt
import sys


# parse argument of main.py function
def parse_args(argv):
    inputfile = ''
    outputfile = ''
    model_path = ''
    run_name = ''
    # baseline_flag = False
    # train_flag = False
    try:
        # print(argv)
        opts, args = getopt.getopt(argv,"hi:o:m:n:",["ifile=","ofile=","mpath=","runname="])
    except:  # getopt.GetoptError:
        print('test.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <inputfile> -o <outputfile> -m <modelpath> -n <runname>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg[0:]
        elif opt in ("-o", "--ofile"):
            outputfile = arg[0:]
        elif opt in ("-m", "--mpath"):
            model_path = arg[0:]
        elif opt in ("-n", "--runname"):
            run_name = arg[0:]
    return inputfile, outputfile, model_path, run_name
