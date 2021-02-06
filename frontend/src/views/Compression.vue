<template>
  <div class="compression_page">
    <h1 class="title">Compressor</h1>
    <div class="comment">Upload facial image to compress</div>
    <div class="upload_area clickable" ref="uploadArea" @click="selectFile">
      <div class="tips">
        <div>Click here to select image to upload</div>
        <div>or drag & drop image here 😊</div>
      </div>
      <input class="none" type="file" accept="image/*" multiple @change="getFile" ref="fileInput" />
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      fileData: [],
    };
  },
  mounted() {
    const dropArea = this.$refs.uploadArea;
    dropArea.addEventListener('drop', this.dropEvent);
    dropArea.addEventListener('dragenter', this.preventFn);
    dropArea.addEventListener('dragover', this.preventFn);
  },
  methods: {
    selectFile() {
      this.$refs.fileInput.click();
    },
    addFiles(files) {
      console.log(files);
      files.forEach((file) => {
        const fileReader = new FileReader();
        fileReader.addEventListener('load', () => {
          this.fileData.push(fileReader.result);
          console.log(this.fileData);
        });
        fileReader.readAsDataURL(file);
      });
    },
    getFile(e) {
      this.addFiles(e.target.files);
    },
    dropEvent(e) {
      this.preventFn(e);
      this.addFiles(e.dataTransfer.files);
    },
    preventFn(e) {
      e.stopPropagation();
      e.preventDefault();
    },
  },
};
</script>

<style lang="scss" scoped>
.compression_page {
  margin: 0 auto;
  padding-top: 60px;
  width: 900px;

  .title {
    margin: 0;
    margin-bottom: 8px;
  }

  .comment {
    color: var(--text2);
    font-size: 20px;
    margin-bottom: 16px;
  }

  .upload_area {
    border: 1px var(--border2) dotted;
    border-radius: 4px;
    height: 450px;

    .tips {
      height: 100%;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      font-size: 32px;
      opacity: 0.3;

      & > div {
        margin-bottom: 12px;
      }
    }
  }
}
</style>
