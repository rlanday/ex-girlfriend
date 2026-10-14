#ifndef SELF_EXTRACT_H 
#define SELF_EXTRACT_H 

#include <stdlib.h>
#include <malloc.h>

#include <string>

struct HeaderInfo {
  int dict_size;
  int new_article_order_size;
  int decomp_input_size;
  // Size of the compressed transformer weights (FX2TFWC2 format, appended
  // as-is and loaded directly — never run through cmix itself).
  int tf_weights_size;
};

void write(const std::string& file_name, HeaderInfo& data) {
  FILE *out = fopen(file_name.c_str() , "wb" );
  fwrite(&data , 1 , sizeof(HeaderInfo) , out );
  fclose(out);
}

void read(const std::string& file_name, HeaderInfo& data) {
  FILE *in = fopen(file_name.c_str() , "rb" );
  fread(&data , 1 , sizeof(HeaderInfo) , in );
  fclose(in);
}


// This function splits the ./cmix binary file into 4 parts:
// 1) actual compressor/decompressor binary
// 2) dictionary (get's it in compressed form and decompresses it)
// 3) new order of articles (get's it in compressed form and decompresses it)
// 4) transformer weights (compressed FX2TFWC2 file, used as extracted)
int selfextract_comp() {
  HeaderInfo header;

// open itslef to read auxilary data (dictionary and neworder)
  FILE *f = NULL, *fo = NULL;
  f = fopen("cmix", "rb");

  // get the size of the whole binary
  fseek(f, 0, SEEK_END);
  size_t fsize = ftell(f);
  fseek(f, 0, SEEK_SET);

  unsigned char *p1 = (unsigned char *)malloc(fsize);
  // read the whole binary to memory
  fread(p1, fsize, 1, f);
  fclose(f);

  // read header info
  fo = fopen("test.dat", "wb");
  memcpy(&header, p1 + fsize - sizeof(HeaderInfo), sizeof(HeaderInfo));
  fwrite(p1 + fsize - sizeof(HeaderInfo), sizeof(HeaderInfo), 1, fo);
  fclose(fo);

  //Remove dictionary if present
  remove(".dict");
  
  size_t decmpressor_binary_size = fsize - header.dict_size - header.new_article_order_size - header.tf_weights_size - sizeof(HeaderInfo);

// produce actual decompressor binary
  fo = fopen(".decomp_bin", "wb");
  fwrite(p1, decmpressor_binary_size, 1, fo);
  fclose(fo);

// produce dictionary and decompress it
  fo = fopen(".dict.comp", "wb");
  fwrite(p1 + decmpressor_binary_size, header.dict_size, 1, fo);
  fclose(fo);


// produce article order and decompress it
  fo = fopen(".new_article_order.comp", "wb");
  fwrite(p1 + decmpressor_binary_size + header.dict_size, header.new_article_order_size, 1, fo);
  fclose(fo);

// produce the transformer weights (already in their loadable compressed
// format, so no decompression subprocess is run on them)
  fo = fopen(".tfweights", "wb");
  fwrite(p1 + decmpressor_binary_size + header.dict_size + header.new_article_order_size, header.tf_weights_size, 1, fo);
  fclose(fo);
//  std::cout << "Decompressing the file with the new article order..." << std::endl;
  system("./cmix -d .new_article_order.comp .new_article_order");

//  std::cout << "Decompressing dictionary..." << std::endl;
  system("./cmix -d .dict.comp .dict");
  free(p1);
  malloc_trim(0);
  return 0;
}

// Same as previous function, but used in decompressor
// This function splits the ./archive9 binary file into 4 parts:
// 1) actual decompressor binary
// 2) dictionary (get's it in compressed form and decompresses it)
// 3) transformer weights (compressed FX2TFWC2 file, used as extracted)
// 4) the cmix-compressed enwik9 payload
int selfextract_decomp() {
  HeaderInfo header;
  FILE *f = NULL, *fo = NULL;
  f = fopen("archive9", "rb");

  fseek(f, 0, SEEK_END);
  size_t fsize = ftell(f);
  fseek(f, 0, SEEK_SET);

  unsigned char *p1 = (unsigned char *)malloc(fsize);
  fread(p1, fsize, 1, f);
  fclose(f);

  // read header info
  fo = fopen("test.dat", "wb");
  fwrite(p1 + fsize - sizeof(HeaderInfo), sizeof(HeaderInfo), 1, fo);
  fclose(fo);
  read("test.dat", header);

  //Remove dictionary if present
  remove(".dict");
  
  size_t decmpressor_binary_size = fsize - header.dict_size - header.tf_weights_size - header.decomp_input_size - sizeof(HeaderInfo);

  fo = fopen(".dict.comp_decomp", "wb");
  fwrite(p1 + decmpressor_binary_size, header.dict_size, 1, fo);
  fclose(fo);

  system("./archive9 -d .dict.comp_decomp .dict");//_decomp

  fo = fopen(".tfweights", "wb");
  fwrite(p1 + decmpressor_binary_size + header.dict_size, header.tf_weights_size, 1, fo);
  fclose(fo);

  fo = fopen(".ready4cmix_decomp", "wb");
  fwrite(p1 + decmpressor_binary_size + header.dict_size + header.tf_weights_size, header.decomp_input_size, 1, fo);
  fclose(fo);

  free(p1);
  malloc_trim(0);
  return 0;
}

#endif // PREPR_H
