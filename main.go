package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/dgraph-io/badger/v2"
	"github.com/pkg/profile"
)

func check(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

var vgraph *badger.DB

//Vamana holds details for performing nn search
type Vamana struct {
	db          *badger.DB
	vocabs      []string
	compVectors [][]uint8
	sqNorms     []float64
	dim         int
	M           int
	R           int
	K           int
	s           uint32
	codewords   [][][]float64
	ds          int
}

var metric = "euclidean"

func main() {
	defer profile.Start(profile.CPUProfile, profile.ProfilePath("./profiling")).Stop()

	// set paths and other variables for the dataset we intend to use
	dim := 300
	M := 10
	R := 20
	K := 256

	vocabCount := 10000

	// read medoid from file
	medoidPath := "../vgraph/glove.6B.300d." + metric + "/medoid.txt"
	dat, err := ioutil.ReadFile(medoidPath)
	check(err)
	medoid, err := strconv.Atoi(string(dat))
	check(err)

	// fmt.Println()
	// fmt.Println("Medoid read from file is: ", medoid)
	// fmt.Println()

	vgraphPath := "../vgraph/glove.6B.300d." + metric + "/db"
	compressedPath := "../vgraph/glove.6B.300d." + metric + "/graph2.txt"

	// open badger instance storing vamana graph
	vgraph, err := badger.Open(badger.DefaultOptions(vgraphPath))
	check(err)
	defer vgraph.Close()

	// create struct to hold Vamana about the index we intend to query.
	// it holds refer,es to the vamana graph, in-memory vocabs and compressed vectors,
	// dimension, max out-degree, and index of the medoid(starting point of every query)
	vam := &(Vamana{
		db:          vgraph,
		vocabs:      make([]string, 0, vocabCount),
		sqNorms:     make([]float64, 0, vocabCount),
		compVectors: make([][]uint8, 0, vocabCount),
		dim:         dim,
		M:           M,
		R:           R,
		K:           K,
		s:           uint32(medoid),
		codewords:   make([][][]float64, M),
		ds:          dim / M,
	})

	// initialize codewords with 0 everywhere
	for m := 0; m < M; m++ {
		cws := make([][]float64, vam.K)
		for k := 0; k < vam.K; k++ {
			cws[k] = make([]float64, vam.ds)
		}
		vam.codewords[m] = cws
	}

	// fmt.Println()
	// fmt.Println(vam.codewords)
	// fmt.Println()

	// open file holding the vocabs and the compressed vectors
	file, err := os.Open(compressedPath)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	// read each line from the file and store the contents (vocabs and compressed vectors)
	// in memory
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {

		parts := strings.Fields(scanner.Text())
		// parts[0] - vocab
		// parts[1] - sq norm
		// parts[2:] - M codeword indices

		vam.vocabs = append(vam.vocabs, parts[0])

		sqn, err := strconv.ParseFloat(parts[1], 64)
		check(err)
		// fmt.Println("square norm is ", sqn)
		vam.sqNorms = append(vam.sqNorms, sqn)

		compVector := make([]uint8, M)

		// fmt.Println("parts 2 onward")
		// fmt.Println(parts[2:])
		// fmt.Println("parts 2 onward")

		for i, val := range parts[2:] {
			num, err := strconv.ParseUint(val, 10, 8) // val in base 10, return should be capable of fitting into 8 bits

			if err != nil {
				log.Fatal("Cannot parse this string as int8: ", val)
			}
			compVector[i] = uint8(num)
		}

		// fmt.Println("compVector being added to compVectors")
		// fmt.Println(compVector)
		// fmt.Println("compVector being added to compVectors")

		vam.compVectors = append(vam.compVectors, compVector)
	}

	// Load codewords
	cwpath := "../vgraph/glove.6B.300d." + metric + "/codewords.txt"
	cwfile, err := os.Open(cwpath)
	check(err)
	defer cwfile.Close()

	scanner = bufio.NewScanner(cwfile)
	for scanner.Scan() {

		parts := strings.Fields(scanner.Text())
		// parts[0] - integer - which of the M subspaces
		// parts[1] - integer - codeword index in [0, 255]
		// parts[2] - float64 - sq norm of codeword
		// parts[3:] - dim/M dimensional codeword

		m, err := strconv.Atoi(parts[0])
		check(err)
		k, err := strconv.Atoi(parts[1])
		check(err)

		t := make([]float64, vam.ds)
		for i, val := range parts[3:] {
			t[i], err = strconv.ParseFloat(val, 64)
			check(err)
		}

		vam.codewords[m][k] = t
	}

	// start := time.Now()
	// result := GetEmbeddingMultiple([]string{"dogn", "king"}, vam)
	// end := time.Now()

	// fmt.Printf("Result is: %+v", result)
	// fmt.Println("Took: ", end.Sub(start))
	// fmt.Println()

	//GET NEAREST NEIGHBORS

	// start := time.Now()
	// closest := vam.GetANN("ghanaian", []float64{}, 10, 16)
	// end := time.Now()

	// fmt.Println()
	// // fmt.Printf("closest: %+v", closest)
	// for _, val := range closest {
	// 	println()
	// 	fmt.Println(val.vocab, " ", val.distance)
	// }
	// fmt.Println("Took: ", end.Sub(start))
	// fmt.Println()

	// CALCULATE STATISTICS
	vam.calcStats()

}

// GraphNode represents node in the vamana graph
type GraphNode struct {
	vocab    string
	vector   []float64
	distance float64
}

func (vam *Vamana) calcStats() {
	n := 10000 //number of vectors to use
	// skip := 0
	//rs[0]: r@1
	rs := make([]float64, 1)

	for i := 0; i < n; i++ {
		closest := vam.GetANN(vam.vocabs[i], []float64{}, 1, 1)
		if closest[0].vocab == vam.vocabs[i] {
			rs[0] = rs[0] + 1
		} else {
			fmt.Println("wrong: ", vam.vocabs[i])
		}
	}
	fmt.Println()
	fmt.Println("Stats using ", n, " vectors")
	fmt.Println("R@1: ", rs[0]/float64(n))
	fmt.Println()
}

// GetANN returns the nearest neighbor
func (vam *Vamana) GetANN(vocab string, xq []float64, k int, beamFactor int) []GraphNode {

	// if vocab is present, we search dataset to find its vector.
	// only the vector can be used for nn query.
	if len(vocab) > 0 {
		res := GetEmbeddingMultiple([]string{vocab}, vam)
		if len(res) > 0 && len(res[0].embedding) == vam.dim {
			xq = res[0].embedding
		} else {
			log.Fatal("The vocab you have specified, ", vocab, " is not in the list of vocabs.")
		}
	}

	// find square norm of xq
	xqSqNorm := sqNorm(xq)

	// construct a lookup table between xq and vam.codewords
	lut := make([][]float64, vam.M)
	for m := 0; m < vam.M; m++ {
		lut[m] = make([]float64, vam.K)
	}

	ds := vam.ds

	for m := 0; m < vam.M; m++ {
		for k := 0; k < vam.K; k++ {

			if metric == "euclidean" {
				lut[m][k] = euclidean(xq[m*ds:(m+1)*ds], vam.codewords[m][k])
			} else {
				lut[m][k] = dotProduct(xq[m*ds:(m+1)*ds], vam.codewords[m][k])
			}
		}
	}

	var lConst int

	if k < 10 {
		lConst = 20 * 5
	} else if k < 20 {
		lConst = 20 * k
	} else if k < 50 {
		lConst = 10 * k
	} else if k < 100 {
		lConst = 5 * k
	} else {
		lConst = 2 * k
	}

	// lConst = 5 * k

	l := []uint32{vam.s}
	v := []uint32{}

	// this keeps the full vectors of nodes we've visited so we can rescore nns
	// this is needed since compressed vecs are used to estimate cosine or euclidean
	// scores during the query.
	visitedNodes := make(map[uint32]GraphNode)

	if lConst < k {
		log.Fatal("lConst cannot be smaller than k")
	}

	lMinusV := minus(l, v)
	for len(lMinusV) > 0 {
		pStar, _ := vam.retainClosestK(beamFactor, lMinusV, xq, xqSqNorm, &lut)

		l = union(l, vam.nout(pStar, &visitedNodes))
		v = union(v, pStar)

		if len(l) > lConst {
			l, _ = vam.retainClosestK(lConst, l, xq, xqSqNorm, &lut)
		}
		lMinusV = minus(l, v)
	}

	// In code segment below, i used the compressed vectors to select the
	// top k from l to return. then i used full vectors to get their actual
	// scores (of those selected k) and rerank them and return.
	// a better approach is done below this code segment

	// closest, _ := d.retainClosestK(k, l, xq)

	// forReturn := make([]GraphNode, 0, len(closest))

	// for _, val := range closest {
	// 	forReturn = append(forReturn, GraphNode{
	// 		vocab:    d.vocabs[val],
	// 		vector:   visitedNodes[val].vector,
	// 		distance: L2(visitedNodes[val].vector, xq),
	// 	})
	// }

	// if metric == "euclidean" {
	// 	sort.Slice(forReturn, func(i, j int) bool {
	// 		return forReturn[i].distance < forReturn[j].distance
	// 	})
	// } else if metric == "cosine" {
	// 	sort.Slice(forReturn, func(i, j int) bool {
	// 		return forReturn[i].distance > forReturn[j].distance
	// 	})
	// }
	// return forReturn

	// A better approach is to use the full vectors in selecting the
	// top k from l, instead of the compressed vectors. This will mean
	// having to sort a larger slice but worth the probable improvement
	// in recall

	forReturn := make([]GraphNode, 0, len(l))

	for _, val := range l {
		forReturn = append(forReturn, GraphNode{
			vocab:    vam.vocabs[val],
			vector:   visitedNodes[val].vector,
			distance: L2(visitedNodes[val].vector, vam.sqNorms[val], xq, xqSqNorm),
		})
	}

	if metric == "euclidean" {
		sort.Slice(forReturn, func(i, j int) bool {
			return forReturn[i].distance < forReturn[j].distance
		})
	} else if metric == "cosine" {
		sort.Slice(forReturn, func(i, j int) bool {
			return forReturn[i].distance > forReturn[j].distance
		})
	}
	return forReturn[0:k]
}

func union(a []uint32, b []uint32) []uint32 {
	for _, val := range b {
		if contains(a, val) == false {
			a = append(a, val)
		}
	}
	return a
}

func minus(s1 []uint32, s2 []uint32) []uint32 {
	if len(s1) == 0 || len(s2) == 0 {
		return s1
	}
	newMs := make([]uint32, 0, len(s1))

	for _, val := range s1 {
		if contains(s2, val) == false {
			newMs = append(newMs, val)
		}
	}
	return newMs
}

func contains(a []uint32, test uint32) bool {
	for _, val := range a {
		if test == val {
			return true
		}
	}
	return false
}

func (vam *Vamana) nout(points []uint32, visitedNodes *map[uint32]GraphNode) []uint32 {

	result := make([]uint32, 0, vam.R*len(points))
	var wg sync.WaitGroup
	var m sync.Mutex

	for _, val := range points {
		wg.Add(1)
		go func(node uint32, result *[]uint32, wg *sync.WaitGroup, m *sync.Mutex) {
			defer wg.Done()

			err := vam.db.View(func(txn *badger.Txn) error {
				item, err := txn.Get([]byte(vam.vocabs[node]))
				if err != nil {
					return err
				}

				err = item.Value(func(val []byte) error {
					vec, nout := vam.decodeVector(val, true)
					m.Lock()
					*result = append(*result, nout...)
					(*visitedNodes)[node] = GraphNode{
						vocab:  vam.vocabs[node],
						vector: vec,
					}
					m.Unlock()
					return nil
				})
				if err != nil {
					return err
				}
				return nil
			})

			if err != nil {
				log.Fatal(err)
			}
		}(val, &result, &wg, &m)
	}

	wg.Wait()
	// sort.Ints(result)
	return result
}

func cosTheta(a []float64, aSqNorm float64, b []float64, bSqNorm float64) float64 {

	var num float64

	for i := 0; i < len(a); i++ {
		num += a[i] * b[i]
	}
	return num / math.Sqrt(aSqNorm*bSqNorm)
}

// L2 implements the L2 norm of two vectors
func L2(a []float64, aSqNorm float64, b []float64, bSqNorm float64) float64 {
	if metric == "cosine" {
		return cosTheta(a, aSqNorm, b, bSqNorm)
	}
	return euclidean(a, b)
}

// L2 is the squared L2 distance between two vectors
func euclidean(a, b []float64) float64 {

	var acc float64 = 0
	for i := 0; i < len(a); i++ {
		s := a[i] - b[i]
		acc += s * s
	}

	// fmt.Println("result from euclidean func")
	// fmt.Println(acc)
	// fmt.Println("result from euclidean func")

	return acc
	// return math.Sqrt(acc)
}

func dotProduct(a, b []float64) float64 {
	var acc float64 = 0
	for i := 0; i < len(a); i++ {
		acc += a[i] * b[i]
	}
	return acc
}

// L2 implements the L2 norm of two vectors
// a is compressed, b is xq (full vector) which waas used in constructing the look up table lut
func (vam *Vamana) compL2(a []uint8, aSqNorm float64, b []float64, bSqNorm float64, lut *[][]float64) float64 {

	// ensure that a has length vam.M
	if len(a) != vam.M {
		log.Fatal("First argument to compL2 must have length ", vam.M, ". Current length is:: ", len(a))
	}

	var acc float64

	if metric == "cosine" {
		for i, val := range a {
			acc += (*lut)[i][val]
		}
		acc /= (aSqNorm * bSqNorm) //technically, I'm supposed to take square root of the denominator but doesn't matter since we're using this result for comparison only

	} else if metric == "euclidean" {
		for i, val := range a {
			acc = acc + (*lut)[i][val]
		}
	}
	// fmt.Println("Hell is =")
	// fmt.Println(len(a))
	// fmt.Println("hell is...")
	return acc
}

// func compEuclidean(a, b []float64) float64 {
// 	var acc float64 = 0
// 	for i := 0; i < len(a); i++ {
// 		s := a[i] - b[i]
// 		acc += s * s
// 	}
// 	return acc
// }

func (vam *Vamana) retainClosestK(count int, candidates []uint32, xq []float64, xqSqNorm float64, lut *[][]float64) ([]uint32, []float64) {
	// fmt.Println("Retaining count: ", count)
	closest := make([]uint32, 0, count)
	distances := make([]float64, 0, count)

	for _, cand := range candidates {

		xCand := vam.compVectors[cand]
		xCandSqNorm := vam.sqNorms[cand]

		candDist := vam.compL2(xCand, xCandSqNorm, xq, xqSqNorm, lut)

		if len(closest) < count {
			closest = append(closest, cand)
			distances = append(distances, candDist)
		} else {

			var replaceableIndex int
			foundReplaceable := false

			for idx, val := range distances {
				var cond1 bool

				if metric == "euclidean" {
					cond1 = candDist < val
				} else if metric == "cosine" {
					cond1 = candDist > val
				}

				if cond1 {
					if foundReplaceable == false {
						foundReplaceable = true
						replaceableIndex = idx
					} else {
						// A replaceable item has previously been found.
						// Replace that replaceable with this newly found replaceable
						// only if the just found replaceable is even farther from xq than
						// the existing replaceable

						var cond2 bool

						if metric == "euclidean" {
							cond2 = distances[idx] > distances[replaceableIndex]
						} else if metric == "cosine" {
							cond2 = distances[idx] < distances[replaceableIndex]
						}

						if cond2 {
							replaceableIndex = idx
						}
					}
				}
			}

			if foundReplaceable == true {
				closest[replaceableIndex] = cand
				distances[replaceableIndex] = candDist
			}
		}
	}
	return closest, distances
}

// GetEmbeddingMultiple gets the embedding of several vocabs at once
func GetEmbeddingMultiple(query []string, z *Vamana) []GetEmbeddingResult {
	// fmt.Println("query length is: ", len(query))
	result := make([]GetEmbeddingResult, len(query))
	var m sync.Mutex
	var wg sync.WaitGroup

	for i, val := range query {
		wg.Add(1)
		go GetEmbedding(val, i, z, &result, &m, &wg)
	}

	wg.Wait()
	return result
}

// GetEmbeddingResult is return type when GetEmbedding is called
type GetEmbeddingResult struct {
	query     string
	embedding []float64
	err       error
}

// GetEmbedding returns embedding of the query word
// returns (wordFound, query, embedding, errorThatOccured
func GetEmbedding(query string, queryIndex int, z *Vamana, result *[]GetEmbeddingResult, m *sync.Mutex, wg *sync.WaitGroup) {
	defer wg.Done()
	var vector []float64

	err := z.db.View(func(txn *badger.Txn) error {
		item, err := txn.Get([]byte(query))
		if err != nil {
			return err
		}

		err = item.Value(func(val []byte) error {
			vector, _ = z.decodeVector(val, false) //false: don't decode nout also
			return nil
		})
		if err != nil {
			return err
		}
		return nil
	})

	if err != nil {

		m.Lock()
		(*result)[queryIndex] = GetEmbeddingResult{err: err, query: query, embedding: nil}
		m.Unlock()
		return
	}

	m.Lock()
	(*result)[queryIndex] = GetEmbeddingResult{err: nil, query: query, embedding: vector}
	m.Unlock()
}

func sqNorm(a []float64) float64 {
	var acc float64
	for _, val := range a {
		acc += val * val
	}
	return acc
}

func (vam Vamana) decodeVector(valBytes []byte, decodeNoutToo bool) ([]float64, []uint32) {
	vectorBytes := valBytes[:8*vam.dim]
	vector := make([]float64, vam.dim)
	for i := 0; i < vam.dim; i++ {
		vector[i] = math.Float64frombits(binary.BigEndian.Uint64(vectorBytes[8*i : 8*(i+1)]))
	}

	if decodeNoutToo == false {
		return vector, nil
	}

	noutBytes := valBytes[8*vam.dim:]
	nout := make([]uint32, 0, vam.R)
	for i := 0; i < (len(noutBytes) / 4); i++ {
		nout = append(nout, binary.BigEndian.Uint32(noutBytes[4*i:4*(i+1)]))
	}
	return vector, nout
}
