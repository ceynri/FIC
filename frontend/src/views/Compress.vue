<template>
  <section class="compression_page">
    <header class="title_wrapper">
      <h1 class="title">Compressor</h1>
      <div class="comment">Upload facial image to compress</div>
    </header>
    <Uploader />
  </section>
</template>

<script>
import Uploader from '@/components/Uploader.vue';

export default {
  components: { Uploader },
};
</script>

<style lang="scss" scoped>
.compression_page {
  margin: 0 auto;
  padding: 60px 0;
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
}
</style>
