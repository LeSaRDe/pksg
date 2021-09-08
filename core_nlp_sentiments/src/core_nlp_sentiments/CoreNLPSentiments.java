package core_nlp_sentiments;

//import com.google.protobuf.Internal;

import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.trees.*;
import edu.stanford.nlp.util.CoreMap;
import org.ejml.simple.SimpleMatrix;

//import java.io.PrintStream;
//import java.text.DecimalFormat;
//import java.text.NumberFormat;
import java.io.FileWriter;
import java.io.IOException;
import java.io.StringWriter;
import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.json.*;
import java.sql.*;


/********************************************************************************
 * NOTES
 * 1. By default, CoreNLP uses PTBLexer. It happens that some characters cannot
 * be recognized by PTBLexer. In such cases, we can customize PTBLexer's behavior
 * in two different ways: first, delete the unknown character; and second, keep
 * it as an individual token. To be clarified, in the first way, the positions of
 * all other tokens will not be affected by any deletion of unknown character.
 * Instead, we will see a "positional gap" between tokens if any deletion in
 * between.
 * "tokenize.options untokenizable=allDelete" is our current configuration. And
 * it means that we delete all unknown characters and log all of them. 
 ********************************************************************************/


public class CoreNLPSentiments {
    /********************************************************************************
     * Inner Classes
     ********************************************************************************/
    private class SentimentNode {
        public int m_snode_id = -1;
        public String m_snode_pos = "";
        public ArrayList<Float> m_snode_sentiments = null;
        public SentimentToken m_snode_token = null;

        public SentimentNode(int snode_id, String snode_pos, ArrayList<Float> snode_sentiments,
                             SentimentToken snode_token) {
            m_snode_id = snode_id;
            m_snode_pos = snode_pos;
            m_snode_sentiments = snode_sentiments;
            m_snode_token = snode_token;
        }

        @Override
        public String toString() {
            return Integer.toString(m_snode_id) + "|" + m_snode_pos + "|" + m_snode_sentiments.toString()
                    + "|" + (m_snode_token == null ? "NULL" : m_snode_token.toString());
        }
    }

    private class SentimentToken {
        public String m_token_str = "";
        // m_token_start is the start character position of this token.
        // m_token_end is the character position right after the last character of the token.
        // e.g. in the sentence "Wearing masks is saving lives.", the token "masks" starts at 8 and ends at 13.
        // the smallest position is 0, and the largest position is equal to the length of the sentence.
        public int m_token_start = -1;
        public int m_token_end = -1;

        public SentimentToken(String token_str, int token_start, int token_end) {
            m_token_str = token_str;
            m_token_start = token_start;
            m_token_end = token_end;
        }

        @Override
        public String toString() {
            return m_token_str + "|" + m_token_start + "|" + m_token_end;
        }
    }

    private class SNodeTuple {
        public Tree m_snode;
        public int m_snode_id;

        public SNodeTuple(Tree snode, int snode_id) {
            m_snode = snode;
            m_snode_id = snode_id;
        }
    }

    private class SentimentEdge {
        public int m_src_snode_id;
        public int m_trg_snode_id;

        public SentimentEdge(int src_snode_id, int trg_snode_id) {
            m_src_snode_id = src_snode_id;
            m_trg_snode_id = trg_snode_id;
        }

        @Override
        public String toString() {
            return Integer.toString(m_src_snode_id) + "->" + Integer.toString(m_trg_snode_id);
        }
    }

    private class SentimentGraph {
        public ArrayList<SentimentNode> m_l_snodes;
        public ArrayList<SentimentEdge> m_l_sedges;

        public SentimentGraph(ArrayList<SentimentNode> l_snodes, ArrayList<SentimentEdge> l_sedges) {
            m_l_snodes = l_snodes;
            m_l_sedges = l_sedges;
        }

        @Override
        public String toString() {
            return "Nodes: " + m_l_snodes.toString() + "\n" + "Edges: " + m_l_sedges.toString();
        }
    }

    /********************************************************************************
     * Memeber Variables
     ********************************************************************************/
    private StanfordCoreNLP m_tokenizer = null;
    private StanfordCoreNLP m_sentiment_pipeline = null;
    private Logger m_logger = null;

    /********************************************************************************
     * Memeber Functions
     ********************************************************************************/
    public CoreNLPSentiments() {
        m_tokenizer = create_tokenizer();
        m_sentiment_pipeline = create_sentiment_pipeline();
        m_logger = Logger.getLogger(Logger.GLOBAL_LOGGER_NAME);
        m_logger.setLevel(GlobalSettings.LOG_LEVEL);
    }

    private StanfordCoreNLP create_tokenizer() {
        Properties tokenizerProps = new Properties();
        tokenizerProps.setProperty("annotators", "tokenize, ssplit");
        tokenizerProps.setProperty("tokenize.options", "untokenizable=allDelete");
        return new StanfordCoreNLP(tokenizerProps);
    }

    private StanfordCoreNLP create_sentiment_pipeline() {
        Properties pipelineProps = new Properties();
        pipelineProps.setProperty("annotators", "parse, sentiment");
        pipelineProps.setProperty("parse.binaryTrees", "true");
        pipelineProps.setProperty("parse.buildgraphs", "false");
        pipelineProps.setProperty("parse.nthreads", String.valueOf(10));
        pipelineProps.setProperty("enforceRequirements", "false");
        return new StanfordCoreNLP(pipelineProps);
    }

    private ArrayList<CoreMap> annotate_text(String text) throws Exception {
        Annotation annotation = m_tokenizer.process(text);
        m_sentiment_pipeline.annotate(annotation);
        ArrayList<CoreMap> l_annotated_sentences =
                (ArrayList<CoreMap>) annotation.get(CoreAnnotations.SentencesAnnotation.class);
        if (l_annotated_sentences.size() <= 0) {
            m_logger.log(Level.SEVERE, "[annotate_text] Invalid text: " + text);
            return null;
        }
        return l_annotated_sentences;
    }

    private ArrayList<SentimentToken> get_tokens_from_sentence(CoreMap annotated_sentence) throws Exception {
        if (annotated_sentence == null) {
            throw new Exception("[get_tokens_from_sentence] annotated_sentence is null.");
        }
        ArrayList<CoreLabel> l_annotated_tokens =
                (ArrayList<CoreLabel>) annotated_sentence.get(CoreAnnotations.TokensAnnotation.class);
        if (l_annotated_tokens.size() <= 0) {
            throw new Exception("[get_tokens_from_sentence] No valid token.");
        }
        ArrayList<SentimentToken> l_stokens = new ArrayList<SentimentToken>();
        for (CoreLabel annotated_token : l_annotated_tokens) {
            String token_str = annotated_token.value();
            int token_start = annotated_token.beginPosition();
            int token_end = annotated_token.endPosition();
            l_stokens.add(new SentimentToken(token_str, token_start, token_end));
        }
        return l_stokens;
    }

    private SentimentGraph get_sentiment_graph_from_annotated_sentence(CoreMap annotated_sentence) throws Exception {
        ArrayList<SentimentToken> l_stokens = get_tokens_from_sentence(annotated_sentence);
        m_logger.log(Level.INFO, "[get_sentiment_graph_from_sentence] l_stokens = " + l_stokens.toString());

        ArrayList<SentimentNode> l_out_snodes = new ArrayList<SentimentNode>();
        ArrayList<SentimentEdge> l_out_sedges = new ArrayList<>();

        Tree sentiment_tree = annotated_sentence.get(SentimentCoreAnnotations.SentimentAnnotatedTree.class);
        ArrayDeque<SNodeTuple> a_parents = new ArrayDeque<SNodeTuple>();
        HashMap<Integer, ArrayDeque<SNodeTuple>> h_childrenlists = new HashMap<>();

        // get the children list for each internal node
        int snode_id = 0;
        a_parents.add(new SNodeTuple(sentiment_tree, snode_id));
        while (!a_parents.isEmpty()) {
            SNodeTuple cur_snode_tup = a_parents.removeFirst();
            Tree cur_snode = cur_snode_tup.m_snode;
            int cur_snode_id = cur_snode_tup.m_snode_id;
            if (cur_snode.depth() > 1) {
                ArrayDeque<SNodeTuple> cur_childrenlist = new ArrayDeque<>();
                for (Tree child : cur_snode.getChildrenAsList()) {
                    snode_id += 1;
                    cur_childrenlist.add(new SNodeTuple(child, snode_id));
                    a_parents.add(new SNodeTuple(child, snode_id));
                }
                h_childrenlists.put(cur_snode_id, cur_childrenlist);
            }
        }
        m_logger.log(Level.INFO, "[get_sentiment_graph_from_sentence] h_childrenlists done.");

        // build SentimentGraph
        Stack<SNodeTuple> s_snodes = new Stack<SNodeTuple>();
        int token_id = 0;
        s_snodes.push(new SNodeTuple(sentiment_tree, 0));
        while (!s_snodes.empty()) {
            SNodeTuple cur_snode_tup = s_snodes.peek();
            Tree cur_snode = cur_snode_tup.m_snode;
            int cur_snode_id = cur_snode_tup.m_snode_id;

            if (h_childrenlists.containsKey(cur_snode_id) && h_childrenlists.get(cur_snode_id).size() > 0) {
                s_snodes.push(h_childrenlists.get(cur_snode_id).getFirst());
            } else {
                Tree done_snode = cur_snode;
                int done_snode_id = cur_snode_id;
                s_snodes.pop();
                if (!s_snodes.empty()) {
                    cur_snode_tup = s_snodes.peek();
                    cur_snode = cur_snode_tup.m_snode;
                    cur_snode_id = cur_snode_tup.m_snode_id;
                }

                SentimentToken expected_done_stoken = null;
                String done_snode_pos = done_snode.label().value();
                SimpleMatrix done_snode_sentiment_vect = RNNCoreAnnotations.getPredictions(done_snode);
                ArrayList<Float> done_snode_sentiments = new ArrayList<Float>();
                for (int i = 0; i < done_snode_sentiment_vect.getNumElements(); i++) {
                    done_snode_sentiments.add((float) done_snode_sentiment_vect.get(i));
                }

                if (done_snode.depth() == 1) {
                    if (done_snode.getChildrenAsList().size() != 1) {
                        throw new Exception("[get_sentiment_graph_from_sentence] Invalid depth-1 snode: "
                                + Integer.toString(done_snode_id));
                    }
                    Tree done_leaf = done_snode.getChildrenAsList().get(0);
                    expected_done_stoken = l_stokens.get(token_id);
                    if (!done_leaf.label().value().equals(expected_done_stoken.m_token_str)) {
                        throw new Exception("[get_sentiment_graph_from_sentence] Mismatching stoken: "
                                + done_leaf.label().value() + " -> " + expected_done_stoken.m_token_str);
                    }
                    token_id += 1;
                } else if (done_snode.depth() < 1) {
                    throw new Exception("[get_sentiment_graph_from_sentence] Expectedly encountered a leaf: "
                            + cur_snode.label().value());
                }

                l_out_snodes.add(new SentimentNode(done_snode_id, done_snode_pos, done_snode_sentiments,
                        expected_done_stoken));
                if (cur_snode_id != done_snode_id) {
                    l_out_sedges.add(new SentimentEdge(cur_snode_id, done_snode_id));
                }

                if (!s_snodes.empty()) {
                    SNodeTuple done_child = h_childrenlists.get(cur_snode_id).removeFirst();
                    int done_child_id = done_child.m_snode_id;
                    if (done_child_id != done_snode_id) {
                        throw new Exception("[get_sentiment_graph_from_sentence] Removed wrong child: "
                                + Integer.toString(done_child_id) + " by " + Integer.toString(done_snode_id) +
                                " from " + Integer.toString(cur_snode_id));
                    }
                }
            }
        }
        m_logger.log(Level.INFO, "[get_sentiment_nodes_from_sentence] Output SentimentGraph: "
                + l_out_snodes.toString() + " | " + l_out_sedges.toString());
        return new SentimentGraph(l_out_snodes, l_out_sedges);
    }

    public ArrayList<SentimentGraph> get_sentiment_graphs_from_text(String text) throws Exception {
        ArrayList<CoreMap> l_annotated_sentence = annotate_text(text);
        if (l_annotated_sentence == null) {
            return null;
        }
        ArrayList<SentimentGraph> l_sgraph = new ArrayList<>();
        for (CoreMap annotated_sentence : l_annotated_sentence) {
            SentimentGraph sgraph = get_sentiment_graph_from_annotated_sentence(annotated_sentence);
            l_sgraph.add(sgraph);
        }
        return l_sgraph;
    }

    public String sgraph_to_json_str(SentimentGraph sgraph) throws IOException {
        JsonObjectBuilder sgraph_json_builder = Json.createObjectBuilder();
        ArrayList<SentimentNode> l_snodes = sgraph.m_l_snodes;
        ArrayList<SentimentEdge> l_sedges = sgraph.m_l_sedges;

        // add nodes to JSON
        JsonArrayBuilder snode_array_builder = Json.createArrayBuilder();
        for (SentimentNode snode : l_snodes) {
            int snode_id = snode.m_snode_id;
            String snode_pos = snode.m_snode_pos;
            ArrayList<Float> snode_sentiments = snode.m_snode_sentiments;
            SentimentToken snode_token = snode.m_snode_token;
            String snode_token_str = "";
            int snode_token_start = -1;
            int snode_token_end = -1;
            if (snode_token != null) {
                snode_token_str = snode_token.m_token_str;
                snode_token_start = snode_token.m_token_start;
                snode_token_end = snode_token.m_token_end;
            }

            JsonArrayBuilder snode_sentiments_builder = Json.createArrayBuilder();
            for (float sentiment_score : snode_sentiments) {
                snode_sentiments_builder = snode_sentiments_builder.add(sentiment_score);
            }
            JsonObjectBuilder snode_builder = Json.createObjectBuilder();
            JsonObject snode_json = snode_builder
                    .add("id", snode_id)
                    .add("pos", snode_pos)
                    .add("sentiments", snode_sentiments_builder.build())
                    .add("token_str", snode_token_str)
                    .add("token_start", snode_token_start)
                    .add("token_end", snode_token_end)
                    .build();
            snode_array_builder = snode_array_builder.add(snode_json);
        }
        sgraph_json_builder = sgraph_json_builder.add("nodes", snode_array_builder.build());

        // add edges to JSON
        JsonArrayBuilder sedge_array_builder = Json.createArrayBuilder();
        for (SentimentEdge sedge : l_sedges) {
            int sedge_src_id = sedge.m_src_snode_id;
            int sedge_trg_id = sedge.m_trg_snode_id;

            JsonObjectBuilder sedge_builder = Json.createObjectBuilder();
            JsonObject sedge_json = sedge_builder
                    .add("src_id", sedge_src_id)
                    .add("trg_id", sedge_trg_id)
                    .build();
            sedge_array_builder = sedge_array_builder.add(sedge_json);
        }
        sgraph_json_builder = sgraph_json_builder.add("edges", sedge_array_builder.build());

        // write JSON to String
        JsonObject sgraph_json = sgraph_json_builder.build();
        return sgraph_json.toString();
    }

    public ArrayList<String> text_to_sgraph_json_strs(String sentence) throws Exception {
        ArrayList<SentimentGraph> l_sgraph = get_sentiment_graphs_from_text(sentence);
        if (l_sgraph == null || l_sgraph.size() <= 0) {
            return null;
        }
        ArrayList<String> l_sgraph_json_str = new ArrayList<>();
        for (SentimentGraph sgraph : l_sgraph) {
            String sgraph_json_str = sgraph_to_json_str(sgraph);
            l_sgraph_json_str.add(sgraph_json_str);
        }
        return l_sgraph_json_str;
    }


    /********************************************************************************
     * TEST ONLY START
     ********************************************************************************/

//    public static ArrayList<String> l_texts = new ArrayList<>(Arrays.asList(
//            "Wearing single layer mask may not be helpful.",
//            "wear help"
//    ));
//
//    enum Output
//    {
//        PENNTREES, VECTORS, ROOT, PROBABILITIES
//    }
//
//    private static final NumberFormat NF = new DecimalFormat("0.0000");
//
//    private static int setIndexLabels(Tree tree, int index)
//    {
//        if (tree.isLeaf())
//        {
//          return index;
//        }
//
//        tree.label().setValue(Integer.toString(index));
//        index++;
//        for (Tree child : tree.children())
//        {
//          index = setIndexLabels(child, index);
//        }
//        return index;
//    }
//
//    private static int outputTreeScores(PrintStream out, Tree tree, int index)
//    {
//        if (tree.isLeaf()) {
//          return index;
//        }
//        out.print("  " + index + ':');
//        SimpleMatrix vector = RNNCoreAnnotations.getPredictions(tree);
//        for (int i = 0; i < vector.getNumElements(); ++i) {
//          out.print("  " + NF.format(vector.get(i)));
//        }
//        out.println();
//        index++;
//        for (Tree child : tree.children()) {
//          index = outputTreeScores(out, child, index);
//        }
//        return index;
//    }
//
//    private static int outputTreeVectors(PrintStream out, Tree tree, int index)
//    {
//        if (tree.isLeaf()) {
//          return index;
//        }
//
//        out.print("  " + index + ':');
//        SimpleMatrix vector = RNNCoreAnnotations.getNodeVector(tree);
//        for (int i = 0; i < vector.getNumElements(); ++i) {
//          out.print("  " + NF.format(vector.get(i)));
//        }
//        out.println();
//        index++;
//        for (Tree child : tree.children()) {
//          index = outputTreeVectors(out, child, index);
//        }
//        return index;
//    }
//
//    private static void setSentimentLabels(Tree tree)
//    {
//        if (tree.isLeaf()) {
//          return;
//        }
//
//        for (Tree child : tree.children()) {
//          setSentimentLabels(child);
//        }
//
//        Label label = tree.label();
//        if (!(label instanceof CoreLabel)) {
//          throw new IllegalArgumentException("Required a tree with CoreLabels");
//        }
//        CoreLabel cl = (CoreLabel) label;
//        cl.setValue(Integer.toString(RNNCoreAnnotations.getPredictedClass(tree)));
//    }
//
//    private static void outputTree(PrintStream out, CoreMap sentence, List<Output> outputFormats)
//    {
//        Tree tree = sentence.get(SentimentCoreAnnotations.SentimentAnnotatedTree.class);
//        for (Output output : outputFormats) {
//          switch (output) {
//          case PENNTREES: {
//            Tree copy = tree.deepCopy();
//            setSentimentLabels(copy);
//            out.println(copy);
//            break;
//          }
//          case VECTORS: {
//            Tree copy = tree.deepCopy();
//            setIndexLabels(copy, 0);
//            out.println(copy);
//            outputTreeVectors(out, tree, 0);
//            break;
//          }
//          case ROOT: {
//            out.println("  " + sentence.get(SentimentCoreAnnotations.SentimentClass.class));
//            break;
//          }
//          case PROBABILITIES: {
//            Tree copy = tree.deepCopy();
//            setIndexLabels(copy, 0);
//            out.println(copy);
//            outputTreeScores(out, tree, 0);
//            break;
//          }
//          default:
//            throw new IllegalArgumentException("Unknown output format " + output);
//          }
//        }
//    }
//
//    public static void main(String[] args)
//    {
//        List<Output> outputFormats = new ArrayList(Arrays.asList(Output.ROOT, Output.PENNTREES, Output.PROBABILITIES));
//        // set up pipeline & tokenizer properties
//        Properties pipelineProps = new Properties();
//        pipelineProps.setProperty("annotators", "parse, sentiment");
//        pipelineProps.setProperty("parse.binaryTrees", "true");
//        pipelineProps.setProperty("parse.buildgraphs", "false");
//        pipelineProps.setProperty("enforceRequirements", "false");
//        Properties tokenizerProps = new Properties();
//        tokenizerProps.setProperty("annotators", "tokenize, ssplit");
//        // build tokenizer
//        StanfordCoreNLP tokenizer = new StanfordCoreNLP(tokenizerProps);
//        // build pipeline
//        StanfordCoreNLP pipeline = new StanfordCoreNLP(pipelineProps);
//        // create a document object
//        for (String sentence : l_texts)
//        {
//            Annotation annotation = tokenizer.process(sentence);
//            pipeline.annotate(annotation);
//            for (CoreMap map : annotation.get(CoreAnnotations.SentencesAnnotation.class))
//            {
//                outputTree(System.out, map, outputFormats);
//            }
//        }
//    }


    public static void main(String[] args) {
        CoreNLPSentiments corenlp_sentiment_ins = new CoreNLPSentiments();
        String test_sentence = "wear help";
        String sgraph_json_file_path = "sgraph_test3.json";
        try {
            ArrayList<String> l_sgraph_json_str = corenlp_sentiment_ins.text_to_sgraph_json_strs(test_sentence);
            System.out.println(l_sgraph_json_str);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /********************************************************************************
     * TEST ONLY END
     ********************************************************************************/
}
