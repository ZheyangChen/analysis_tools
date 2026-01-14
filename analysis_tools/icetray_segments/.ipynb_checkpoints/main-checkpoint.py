from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    
    parser.add_argument("-i", "--infile",type=str,
                          default="",
                          dest="INFILE",
                          help="read input physcis to INFILE (.i3.bz2 format)")
    
    parser.add_argument("-o", "--outfile",type=str,
                          default="",
                          dest="OUTFILE",
                          help="output file name")
    
    parser.add_argument("-g", "--gcd",type=str,
                          default="",
                          dest="GCD",
                          help="geometry file")
    
    args = parser.parse_args ()
    
    infile = args.INFILE
    outfile = args.OUTFILE
    GEO = args.GCD
    if os.path.exists(infile) == False:
        print('***file does not exist:')
        sys.exit()
    
    
    tray = I3Tray()
    
    infiles = []
    infiles.append(GEO)
    infiles.append(infile)
    
    tray.AddModule("I3Reader","reader",FileNameList = infiles)
    
    
    tray.AddModule('I3Writer',
                   'writer',
                   filename=args.out,
                   streams=[icetray.I3Frame.TrayInfo,
                            icetray.I3Frame.Physics,
                            icetray.I3Frame.Simulation,
                            icetray.I3Frame.Stream('M'),
                            icetray.I3Frame.DAQ])
    
    if args.nframes is None:
        tray.Execute()
    else:
        tray.Execute(args.nframes)
    tray.PrintUsage()


if __name__ == '__main__':
    logging.basicConfig(encoding='utf-8', level=logging.INFO)
    main()




