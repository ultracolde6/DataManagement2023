import numpy as np

def process_timer_py(bindata):
    channels = 7 # assumes each bit other than the channels is part of the time
    bitstot = 32
    clktime = 8.298/10**6 # clock cycle duration, in ms
    # tparts = 2**np.arange(bitstot-channels) #BAD horrible UCHIcago code
    tparts = np.concatenate((np.flip(2**np.arange(8)),np.flip(2**np.arange(8,16)),np.flip(2**np.arange(16,24)),[16777216])) #GOOD amazing Cal code
    unit_time = 1/700 #us
    PHOTON_TIMERS = ['PT_', 'PT2_', 'PT3_']
    PT_channels = np.array([7,7,7])

    default_superChannelNames = [['SuperChannel1', 'A Channel Name', 'A Channel Name', 'A Channel Name', 'A Channel Name', 'A Channel Name', 'A Channel Name'],\
                                ['SuperChannel2',], ['SuperChannel3',]]
    default_assignments = [[0,1,2,3,4,5,6], [0,]*7, [0,]*7]
    PT_delays = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6]])

    assignments = {}
    superchannels = {}
    superChannelNames = {}
    # DETERMINE HOW MANY DIFFERENT TIMERS ARE IN USE
    TIMER_FLAGS = [False,]*len(PHOTON_TIMERS) #false(length(PHOTON_TIMERS), 1);
    sc_names_all = []
    sc_counter = 0

    which = {}
    which_ch = {}
    thischannels = {}
    delays_sorted = {}
    delay_sorter = {}
    sc_names_all = {}
    which = {}
    thischannels = {}
    for timer in range(len(PHOTON_TIMERS)):
        TIMER_FLAGS[timer] = True;

        superChannelNames[timer] = default_superChannelNames[timer]
        assignments[timer] = default_assignments[timer]
        superchannels[timer] = np.unique(assignments[timer])


        # collect all of the superchannel names and the total number
        # as well as other useful information for assigning to
        # superchannels later

        for sx in range(len(superChannelNames[timer])):
            sc_names_all[sc_counter] = superChannelNames[timer][sx]
            which[sc_counter] = assignments[timer]==superchannels[timer][sx] # which channels are in this superchannel
            which_ch[sc_counter] = np.nonzero(which[sc_counter])[0]

            thischannels[sc_counter] = sum(which[sc_counter])

            delays = PT_delays[timer, which[sc_counter]]
            delays_sorted[sc_counter] = np.sort(delays)
            delay_sorter[sc_counter] = np.argsort(delays)
            sc_counter +=1

    #TODO fix
    tix = 0 #timer boxes
    ix = 0 #uv
    jx = 0 #file

    possible = np.arange(len(PHOTON_TIMERS))
    events_sc = [[] for i in range(sc_counter)]
    events_sc_malformed = []

    CLK_dts = []
    # counter to keep track of how many SC have already been
    # accounted for by previous timers
    sc_counter_sofar = 0
    skipflag = False # this gets set to true when the next event has already been interpreted as part of a superchannel event
    for kx in range(len(bindata)//bitstot): # loop through each 32 bit event within the current file
        # -turns out there are occassional bullshit events
        # which are all zeros, in the middle of lots of
        # useful data; so now we check for THREE empty
        # events in a row before we break out
        # -ALSO ALSO, note that we have a bit assigned to
        # each channel, so that an event of all zeros NEVER
        # GETS ASSIGNED TO ANY CHANNEL
        # -if we need 8 channels someday, we can assign that
        # to all zeros on the channel bits... and then just
        # ignore time = 0
        if np.sum(bindata[bitstot*kx:bitstot*(kx+1)-channels]) == 0:
            if kx<len(bindata)//bitstot-1 and np.sum(bindata[bitstot*(kx+1):bitstot*(kx+2)-channels]) == 0:
                if kx<len(bindata)//bitstot-2 and np.sum(bindata[bitstot*(kx+2):bitstot*(kx+3)-channels]) == 0:
                    #print("Three zeros in a row detected, break")
                    break

        clock_cycle = np.sum(tparts*bindata[bitstot*kx:bitstot*(kx+1)-channels])

        # Here we assign events to superchannels
        if skipflag: # then this event was already interpreted as part of the previous event
            skipflag = False
        else:
            chdata = bindata[bitstot*(kx+1)-channels:bitstot*(kx+1)]

            for sx in range(len(superchannels[tix])):
                sx_full = sx + sc_counter_sofar
                chthis = chdata[which_ch[sx_full][delay_sorter[sx_full]]]

                if sum(chthis) == len(chthis): # all channels triggered simultaneously THIS IS A WEIRD STATEMENT SINCE chthis IS SCALAR?
                    # in this case, assign earliest
                    # oversampled clock cycle
                    clock_cycle_sc = PT_channels[tix]*(clock_cycle-1)+1;


                elif kx<len(bindata)//bitstot-1: # make sure you don't exceed data length looking for next channel
                    clock_cycle_next = np.sum(tparts*bindata[bitstot*(kx+1):bitstot*(kx+2)-channels])

                    if clock_cycle_next == clock_cycle + 1:
                        skipflag = True # next event will be considered part of the same superevent
                        chdata_next = bindata[bitstot*(kx+2)-channels:bitstot*(kx+2)]
                        chthis_next = chdata_next[which_ch[sx_full][delay_sorter[sx_full]]]

                        # no event in this superchannel,
                        # ignore
                        if np.sum(chthis) + np.sum(chthis_next) == 0:
                            continue

                        if np.sum(chthis_next)==len(chthis): # all channels triggered simultaneously in NEXT clock cycle
                            clock_cycle_sc = PT_channels[tix]*(clock_cycle_next-1)+1
                        elif np.sum(chthis) + np.sum(chthis_next) != len(chthis) or \
                            np.max(chthis+chthis_next)> 1 or np.where(chthis_next!=0)[0][0] < np.where(chthis!=0)[0][-1]: # event is malformed; store clock cycle,
                            clock_cycle_sc = PT_channels[tix]*(clock_cycle-1)+1
                            events_sc_malformed.append((clock_cycle_sc, (chthis+2*chthis_next), jx))
                            continue
                        else:
                            # Event should be well-formed,
                            # AND contains triggers in both
                            # this clock cycle and the
                            # next.
                            # Set proper oversampled clock
                            # cycle
                            clock_cycle_sc = PT_channels[tix]*(clock_cycle)+1-delays_sorted[sx_full][np.where(chthis_next!=0)[0][0]]



                    elif sum(chthis) == 0: # no event in this superchannel, ignore
                        continue
                    else:  # event is malformed; store clock cycle, which channels triggered, and run number
                        clock_cycle_sc = PT_channels[tix]*(clock_cycle-1)+1
                        events_sc_malformed.append((clock_cycle_sc, (chthis+2*chthis_next), jx))
                        continue

                # should only arrive here for well-formed
                # events; should have assigned
                # clock_cycle_sc already as well
                events_sc[sx_full].append(clock_cycle_sc)

    return events_sc, events_sc