#!/usr/bin/env python3


def get_tree_from_point(mesh, index):

    # Get our parts
    colour = mesh[0]
    stencil = mesh[1]
    graph = mesh[2]

    # Some constants
    n_points = 200

    # Get 200 points centred around our input index
    queue = collections.deque([0, index])
    points = set()

    # We only actually gather 199, since 0 is the null point
    while len(queue) > 0 and len(points) < n_points:
        # Get our next element and add it to our points
        i = queue.popleft()
        points.add(i)

        # Add our points to the queue but don't bother checking the first one (that's us)
        for n in graph[i][1:]:
            if n not in points and n not in queue:
                queue.append(n)

    # Sort and make into a list
    points = sorted(list(points))

    # If we don't have enough points add 0s on the end until we do
    if len(points) < 200:
        points.extend([0] * (200 - len(points)))

    points = np.array(points)

    # Make our reverse lookup map
    r_indices = np.zeros(len(stencil), dtype=np.int32)
    for i, v in enumerate(points):
        r_indices[v] = i

    # Gather our new data
    new_graph = np.take(r_indices, np.take(graph, points, axis=0))
    new_colour = np.take(colour, points, axis=0)
    new_stencil = np.take(stencil, points, axis=0)

    # Return what we learnt
    return r_indices[index], new_graph, new_colour, new_stencil


def resample(sess, network, in_dir, out_dir):

    # Minimum chance of being selected is 5%
    min_chance = 0.05

    # Get our data packs
    packs = sorted([f for f in os.listdir(in_dir) if f.endswith('.bin.lz4')])

    # Work through each pack
    for pack in packs:
        print('Loading data tree pack {}'.format(pack))
        data = load.pack(os.path.join(in_dir, pack))
        print('\tloaded')

        samples = []

        for in_X, in_Y, in_G in zip(*data):

            # Run our network for this input object
            result, = sess.run([network['network']], feed_dict={
                network['X']: [in_X],
                network['G']: [in_G],
                network['K']: 1.0
            })

            # We are not running batches so this is always 1 element
            result = result[0]

            # Now we must select all the ball points
            bp, = np.nonzero(in_Y)

            # Now get the probabilities of selecting each point
            # The more wrong the values are the more likely they are to be selected
            probs = np.abs(result[:,0] - in_Y) + (min_chance / (len(in_Y) - len(bp)))

            # We will never reselect our ball points so set their probability to 0
            np.put(probs, bp, 0.0)
            # Also we can never select the null point
            probs[0] = 0.0

            # Normalise our probabilities so they sum to 1
            probs = probs / np.sum(probs)

            # Randomly select n points
            nbp = np.random.choice(a=len(probs), size=len(bp), p=probs, replace=False)

            # Add them to the list
            for p in bp.tolist() + nbp.tolist():
                samples.append(get_tree_from_point((in_X, in_Y, in_G), p))

        # Save our samples in a batch
        with open(os.path.join(out_dir, os.path.splitext(pack)[0]), 'wb') as f:
            # Write out the data in a random order
            random.shuffle(samples)
            for i, new_graph, new_colour, new_stencil in samples:

                if (len(new_graph) != 200):
                    raise Exception("Oh snap!")

                f.write(struct.pack('<If', i, new_stencil[i]))
                f.write(new_graph.tobytes())
                f.write(new_colour.tobytes())
                f.write(new_stencil.tobytes())
